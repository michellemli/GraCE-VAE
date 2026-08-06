import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.data import Data, Batch
from typing import Optional
from edgeindex import build_candidate_1hop_subgraph


SUPPORTED_GNN_MODES = {3, 4, 5, 6, 11, 13, 15}
# VAE model with causal layer and mmd loss
# "dim" specifies the sample dimension; "c_dim" specifies the dimension of the intervention encoding.
#  "z_dim" specifies the dimension of the latent space.
def _get_target_nodes_from_c(c_row: torch.Tensor, ptb_to_node: torch.Tensor) -> torch.Tensor:
    """c_row: [c_dim] one-hot/multi-hot. Return node ids [k]."""
    ptb_ids = torch.where(c_row > 0.0)[0]
    if ptb_ids.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=c_row.device)
    return ptb_to_node[ptb_ids].long()


def _get_intervention_key(c_row: torch.Tensor) -> tuple:
    """Stable cache key for one intervention row."""
    return tuple(torch.where(c_row > 0.0)[0].detach().cpu().tolist())


class FixedTriValueInterventionEncoder(nn.Module):
    """
    Variant 1
    ---------
    Fixed node-level initialization:
      - other genes: -1
      - candidate (possible intervened) genes: 0
      - current target gene(s): 1
    + low-dim node embedding to learn gene semantics.
    Uses a fixed graph (NO edge deletion).

    Input : c [B, c_dim]
    Output: bc [B, z_dim], eta [B]
    """

    def __init__(
        self,
        num_nodes: int,
        z_dim: int,
        base_edge_index: torch.Tensor,
        ptb_to_node: torch.Tensor,
        node_emb_dim: int = 8,
        gnn_hidden: int = 32,
        mlp_hidden: int = 128,
        readout: str = "target",  # "graph" | "target" | "graph+target"
        candidate_nodes: Optional[torch.Tensor] = None,
        gnn_dropout: float = 0.1,
        use_ptb_strength: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.z_dim = int(z_dim)
        self.readout = readout
        self.gnn_dropout = float(gnn_dropout)
        self.use_ptb_strength = bool(use_ptb_strength)

        self.register_buffer("base_edge_index", base_edge_index)
        self.register_buffer("ptb_to_node", ptb_to_node)
        self.c_dim = int(ptb_to_node.numel())

        # candidate mask
        if candidate_nodes is None:
            cand_nodes = ptb_to_node.unique()
        else:
            cand_nodes = candidate_nodes.long()
        cand_mask = torch.zeros(self.num_nodes, dtype=torch.float)
        cand_mask[cand_nodes] = 1.0
        self.register_buffer("candidate_mask", cand_mask)
        self._target_node_cache = {}

        # low-dim gene context embedding
        self.node_emb = nn.Embedding(self.num_nodes, node_emb_dim)

        # input feature = [tri_value] + node_emb
        in_dim = 1 + node_emb_dim

        self.conv1 = GCNConv(in_dim, gnn_hidden)
        self.conv2 = GCNConv(gnn_hidden, gnn_hidden)
        self.norm1 = nn.LayerNorm(gnn_hidden)
        self.norm2 = nn.LayerNorm(gnn_hidden)

        ro_dim = gnn_hidden if readout in ["graph", "target"] else 2 * gnn_hidden
        self.mlp1 = nn.Linear(ro_dim, mlp_hidden)
        self.mlp2 = nn.Linear(mlp_hidden, z_dim)
        self.eta_head = nn.Linear(mlp_hidden, 1)
        self.eta_scale = nn.Parameter(torch.tensor(1.0))

        # optional: per-ptb scalar baseline
        if self.use_ptb_strength:
            self.ptb_strength = nn.Embedding(self.c_dim, 1)
            nn.init.zeros_(self.ptb_strength.weight)

        self.act = nn.LeakyReLU(0.2)

    def _softmax_temp(self, logits: torch.Tensor, temp: float):
        t = float(temp)
        t = max(min(t, 10.0), 1e-3)
        # match your existing convention: temp bigger => sharper
        return F.softmax(logits * t, dim=0)

    def _encode_one(self, c_row: torch.Tensor, temp: float):
        device = c_row.device
        dtype = self.node_emb.weight.dtype

        key = _get_intervention_key(c_row)
        target_nodes = self._target_node_cache.get(key)
        if target_nodes is None:
            target_nodes = _get_target_nodes_from_c(c_row, self.ptb_to_node)
            self._target_node_cache[key] = target_nodes
        if target_nodes.numel() == 0:
            bc = torch.full((self.z_dim,), 1.0 / self.z_dim, device=device, dtype=dtype)
            eta = torch.tensor(0.0, device=device, dtype=dtype)
            return bc, eta

        # tri-value init
        tri = torch.full((self.num_nodes, 1), -1.0, device=device, dtype=dtype)
        tri[self.candidate_mask.to(device).bool()] = 0.0
        tri[target_nodes] = 1.0

        e = self.node_emb.weight.to(device)
        x = torch.cat([tri, e], dim=-1)

        ei = self.base_edge_index
        h1 = self.conv1(x, ei)
        h1 = self.norm1(self.act(h1))
        h1 = F.dropout(h1, p=self.gnn_dropout, training=self.training)
        h2 = self.conv2(h1, ei)
        h = self.norm2(self.act(h2 + h1))
        h = F.dropout(h, p=self.gnn_dropout, training=self.training)

        g_emb = h.mean(dim=0)
        t_emb = h[target_nodes].mean(dim=0)

        if self.readout == "graph":
            emb = g_emb
        elif self.readout == "target":
            emb = t_emb
        else:
            emb = torch.cat([t_emb, (t_emb - g_emb)], dim=0)

        hh = self.act(self.mlp1(emb))
        logits = self.mlp2(hh)
        bc = self._softmax_temp(logits, temp)

        eta_res = self.eta_scale * torch.tanh(self.eta_head(hh).squeeze(-1))
        if self.use_ptb_strength:
            ptb_ids = torch.where(c_row > 0.0)[0]
            eta_base = self.ptb_strength(ptb_ids).mean().squeeze(-1)
            eta = eta_base + eta_res
        else:
            eta = eta_res

        return bc, eta

    def forward(self, c: torch.Tensor, temp: float = 1.0):
        B = c.size(0)
        # In this project each batch is constructed to share the same intervention.
        bc0, eta0 = self._encode_one(c[0], temp)
        return bc0.unsqueeze(0).repeat(B, 1), eta0.repeat(B)


class FixedTriValueSmallGraphEncoder(nn.Module):
    """
    tri-value on SMALL graph only (nodes already relabeled to 0..N_sub-1)
      - other nodes: -1
      - candidate nodes: 0
      - target nodes: 1
    fixed graph, no edge deletion
    """
    def __init__(
        self,
        num_nodes: int,                 # N_sub
        z_dim: int,
        edge_index: torch.Tensor,       # [2, E_sub] local ids
        ptb_to_node: torch.Tensor,      # [c_dim] local ids
        candidate_nodes: torch.Tensor,  # [<=105] local ids
        node_emb_dim: int = 4,
        gnn_hidden: int = 16,
        mlp_hidden: int = 128,
        readout: str = "target",
        gnn_dropout: float = 0.1,
        use_ptb_strength: bool = True,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.z_dim = int(z_dim)
        self.readout = readout
        self.gnn_dropout = float(gnn_dropout)

        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("ptb_to_node", ptb_to_node.long())
        self.c_dim = int(ptb_to_node.numel())

        cand_mask = torch.zeros(self.num_nodes, dtype=torch.float)
        cand_mask[candidate_nodes.long()] = 1.0
        self.register_buffer("candidate_mask", cand_mask)
        self._target_node_cache = {}

        self.node_emb = nn.Embedding(self.num_nodes, node_emb_dim)

        in_dim = 1 + node_emb_dim
        self.conv1 = GCNConv(in_dim, gnn_hidden)
        self.conv2 = GCNConv(gnn_hidden, gnn_hidden)
        self.norm1 = nn.LayerNorm(gnn_hidden)
        self.norm2 = nn.LayerNorm(gnn_hidden)

        ro_dim = gnn_hidden if readout in ["graph", "target"] else 2 * gnn_hidden
        self.mlp1 = nn.Linear(ro_dim, mlp_hidden)
        self.mlp2 = nn.Linear(mlp_hidden, z_dim)
        self.eta_head = nn.Linear(mlp_hidden, 1)
        self.eta_scale = nn.Parameter(torch.tensor(1.0))

        self.use_ptb_strength = bool(use_ptb_strength)
        if self.use_ptb_strength:
            self.ptb_strength = nn.Embedding(self.c_dim, 1)
            nn.init.zeros_(self.ptb_strength.weight)

        self.act = nn.LeakyReLU(0.2)

    def _softmax_temp(self, logits, temp):
        t = max(min(float(temp), 10.0), 1e-3)
        return F.softmax(logits * t, dim=0)

    def _encode_one(self, c_row, temp):
        device = c_row.device
        dtype = self.node_emb.weight.dtype

        ptb_ids = torch.where(c_row > 0)[0]
        if ptb_ids.numel() == 0:
            bc = torch.full((self.z_dim,), 1.0 / self.z_dim, device=device, dtype=dtype)
            eta = torch.tensor(0.0, device=device, dtype=dtype)
            return bc, eta

        key = _get_intervention_key(c_row)
        target_nodes = self._target_node_cache.get(key)
        if target_nodes is None:
            target_nodes = self.ptb_to_node[ptb_ids].long()
            self._target_node_cache[key] = target_nodes

        tri = torch.full((self.num_nodes, 1), -1.0, device=device, dtype=dtype)
        tri[self.candidate_mask.to(device).bool()] = 0.0
        tri[target_nodes] = 1.0

        x = torch.cat([tri, self.node_emb.weight.to(device)], dim=-1)
        ei = self.edge_index.to(device)

        h1 = self.norm1(self.act(self.conv1(x, ei)))
        h1 = F.dropout(h1, p=self.gnn_dropout, training=self.training)
        h2 = self.conv2(h1, ei)
        h = self.norm2(self.act(h2 + h1))
        h = F.dropout(h, p=self.gnn_dropout, training=self.training)

        g_emb = h.mean(dim=0)
        t_emb = h[target_nodes].mean(dim=0)

        if self.readout == "graph":
            emb = g_emb
        elif self.readout == "target":
            emb = t_emb
        else:
            emb = torch.cat([t_emb, (t_emb - g_emb)], dim=0)

        hh = self.act(self.mlp1(emb))
        bc = self._softmax_temp(self.mlp2(hh), temp)

        eta_res = self.eta_scale * torch.tanh(self.eta_head(hh).squeeze(-1))
        if self.use_ptb_strength:
            eta_base = self.ptb_strength(ptb_ids).mean().squeeze(-1)
            eta = eta_base + eta_res
        else:
            eta = eta_res
        return bc, eta

    def forward(self, c, temp=1.0):
        B = c.size(0)
        # In this project each batch is constructed to share the same intervention.
        bc0, eta0 = self._encode_one(c[0], temp)
        return bc0.unsqueeze(0).repeat(B, 1), eta0.repeat(B)


class DropEdgeDistillInterventionEncoder(nn.Module):
    """
    Variant 2
    ---------
    Keep the original "drop incident edges" intervention encoder, but add an MLP teacher
    (baseline) and distillation to stabilize training.

    Three output modes:
      - output_mode='teacher': return teacher outputs (guaranteed >= baseline performance)
      - output_mode='student': return graph/drop-edge outputs
      - output_mode='mix'    : convex mix of teacher & student with learnable gate

    Distillation loss is stored in self.distill_loss after forward().
    """

    def __init__(
        self,
        num_nodes: int,
        z_dim: int,
        base_edge_index: torch.Tensor,
        ptb_to_node: torch.Tensor,
        node_emb_dim: int = 32,
        gnn_hidden: int = 64,
        mlp_hidden: int = 128,
        teacher_hidden: int = 128,
        readout: str = "target",
        gnn_dropout: float = 0.0,
        output_mode: str = "mix",      # "teacher" | "student" | "mix"
        alpha_init: float = -2.0,       # sigmoid(alpha_init) ~ 0.12 (start close to teacher)
        use_teacher_eta_ptb: bool = True,
        distill_bc: bool = True,
        distill_eta: bool = True,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.z_dim = int(z_dim)
        self.readout = readout
        self.gnn_dropout = float(gnn_dropout)
        self.output_mode = output_mode
        self.use_teacher_eta_ptb = bool(use_teacher_eta_ptb)
        self.distill_bc = bool(distill_bc)
        self.distill_eta = bool(distill_eta)

        self.register_buffer("base_edge_index", base_edge_index)
        self.register_buffer("ptb_to_node", ptb_to_node)
        self.c_dim = int(ptb_to_node.numel())
        self._target_node_cache = {}
        self._dropped_edge_cache = {}

        # ----- student (drop-edge graph encoder) -----
        self.node_emb = nn.Embedding(self.num_nodes, node_emb_dim)
        self.conv1 = GCNConv(node_emb_dim, gnn_hidden)
        self.conv2 = GCNConv(gnn_hidden, gnn_hidden)
        self.norm1 = nn.LayerNorm(gnn_hidden)
        self.norm2 = nn.LayerNorm(gnn_hidden)

        ro_dim = gnn_hidden if readout in ["graph", "target"] else 2 * gnn_hidden
        self.mlp1_s = nn.Linear(ro_dim, mlp_hidden)
        self.mlp2_s = nn.Linear(mlp_hidden, z_dim)
        self.eta_head_s = nn.Linear(mlp_hidden, 1)
        self.eta_scale = nn.Parameter(torch.tensor(1.0))

        # ----- teacher (MLP on c) -----
        self.c1 = nn.Linear(self.c_dim, teacher_hidden)
        self.c2 = nn.Linear(teacher_hidden, z_dim)
        self.eta_head_t = nn.Linear(teacher_hidden, 1)
        self.c_shift = nn.Parameter(torch.zeros(self.c_dim))  # per-ptb strength baseline

        # mix gate
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.act = nn.LeakyReLU(0.2)
        self.distill_loss = None

    def _softmax_temp_vec(self, logits: torch.Tensor, temp: float, dim: int):
        t = float(temp)
        t = max(min(t, 10.0), 1e-3)
        return F.softmax(logits * t, dim=dim)

    def _drop_incident_edges(self, target_nodes: torch.Tensor) -> torch.Tensor:
        ei = self.base_edge_index
        if target_nodes.numel() == 0:
            return ei
        src, dst = ei[0], ei[1]
        if hasattr(torch, "isin"):
            bad = torch.isin(src, target_nodes) | torch.isin(dst, target_nodes)
        else:
            bad = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
            for t in target_nodes.tolist():
                bad |= (src == t) | (dst == t)
        return ei[:, ~bad]

    def _encode_student_one(self, c_row: torch.Tensor, temp: float):
        device = c_row.device
        dtype = self.node_emb.weight.dtype

        key = _get_intervention_key(c_row)
        target_nodes = self._target_node_cache.get(key)
        if target_nodes is None:
            target_nodes = _get_target_nodes_from_c(c_row, self.ptb_to_node)
            self._target_node_cache[key] = target_nodes
        if target_nodes.numel() == 0:
            bc = torch.full((self.z_dim,), 1.0 / self.z_dim, device=device, dtype=dtype)
            eta = torch.tensor(0.0, device=device, dtype=dtype)
            return bc, eta

        ei = self._dropped_edge_cache.get(key)
        if ei is None:
            ei = self._drop_incident_edges(target_nodes)
            self._dropped_edge_cache[key] = ei
        x = self.node_emb.weight.to(device)

        h1 = self.conv1(x, ei)
        h1 = self.norm1(self.act(h1))
        h1 = F.dropout(h1, p=self.gnn_dropout, training=self.training)
        h2 = self.conv2(h1, ei)
        h = self.norm2(self.act(h2 + h1))
        h = F.dropout(h, p=self.gnn_dropout, training=self.training)

        g_emb = h.mean(dim=0)
        t_emb = h[target_nodes].mean(dim=0)
        if self.readout == "graph":
            emb = g_emb
        elif self.readout == "target":
            emb = t_emb
        else:
            emb = torch.cat([t_emb, (t_emb - g_emb)], dim=0)

        hh = self.act(self.mlp1_s(emb))
        logits = self.mlp2_s(hh)
        bc = self._softmax_temp_vec(logits, temp, dim=0)
        eta = self.eta_scale * torch.tanh(self.eta_head_s(hh).squeeze(-1))
        return bc, eta

    def _encode_teacher_batch(self, c: torch.Tensor, temp: float):
        hh = self.act(self.c1(c))
        logits = self.c2(hh)
        bc = self._softmax_temp_vec(logits, temp, dim=1)
        if self.use_teacher_eta_ptb:
            eta = c @ self.c_shift
        else:
            eta = self.eta_head_t(hh).squeeze(-1)
        return bc, eta

    def forward(self, c: torch.Tensor, temp: float = 1.0):
        B = c.size(0)
        device = c.device
        eps = 1e-8

        # teacher is cheap, compute for whole batch
        bc_t, eta_t = self._encode_teacher_batch(c, temp)

        # In this project each batch is constructed to share the same intervention.
        bc0, eta0 = self._encode_student_one(c[0], temp)
        bc_s = bc0.unsqueeze(0).repeat(B, 1)
        eta_s = eta0.repeat(B)

        # distillation loss (teacher -> student)
        distill = torch.tensor(0.0, device=device, dtype=bc_s.dtype)
        if self.distill_bc:
            # KL(teacher || student)
            distill = distill + F.kl_div(torch.log(bc_s + eps), bc_t.detach(), reduction='batchmean')
        if self.distill_eta:
            distill = distill + F.mse_loss(eta_s, eta_t.detach())
        self.distill_loss = distill

        # output selection / mixing
        if self.output_mode == "teacher":
            return bc_t, eta_t
        if self.output_mode == "student":
            return bc_s, eta_s
        if self.output_mode == "mix":
            a = torch.sigmoid(self.alpha)
            bc = (1 - a) * bc_t + a * bc_s
            bc = bc / (bc.sum(dim=1, keepdim=True) + eps)
            eta = (1 - a) * eta_t + a * eta_s
            return bc, eta
        raise ValueError(f"Unknown output_mode: {self.output_mode}")


# ============================================================
# GRACE wrapper using the above encoders
# ============================================================

def weights_init(m):
    # copied from your original model.py
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        truncated_normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


def truncated_normal_(tensor, mean=0, std=0.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class GRACE(nn.Module):
    """A drop-in replacement of your GRACE, but with selectable causal intervention encoders."""

    def __init__(
        self,
        dim,
        z_dim,
        c_dim,
        p_dim,
        mode,
        device=None,
        interv_num_nodes=None,
        interv_edge_index=None,
        ptb_to_node=None,
        node_emb_dim=2,
        gnn_hidden=8,
        readout="target",
        interv_encoder_type: str = "v2_dropedge",
    ):
        super().__init__()
        if mode not in SUPPORTED_GNN_MODES:
            raise ValueError(f"Unsupported GNN mode: {mode}")
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim
        self.p_dim = p_dim

        self.device = 'cpu' if device == 'cpu' else device
        self.cuda = (self.device != 'cpu')

        # always define
        self.graph_c_encoder = None

        if (interv_num_nodes is not None) and (interv_edge_index is not None) and (ptb_to_node is not None):
            if interv_encoder_type == "v1_trivalue":
                self.graph_c_encoder = FixedTriValueInterventionEncoder(
                    num_nodes=interv_num_nodes,
                    z_dim=self.z_dim,
                    base_edge_index=interv_edge_index,
                    ptb_to_node=ptb_to_node,
                    node_emb_dim=node_emb_dim,
                    gnn_hidden=gnn_hidden,
                    mlp_hidden=128,
                    readout=readout,
                    gnn_dropout=0.1,
                    use_ptb_strength=True,
                )
            elif interv_encoder_type == "v2_dropedge":
                # old behaviour (but using the student part only)
                self.graph_c_encoder = DropEdgeDistillInterventionEncoder(
                    num_nodes=interv_num_nodes,
                    z_dim=self.z_dim,
                    base_edge_index=interv_edge_index,
                    ptb_to_node=ptb_to_node,
                    node_emb_dim=max(int(node_emb_dim), 8),
                    gnn_hidden=gnn_hidden,
                    mlp_hidden=128,
                    readout=readout,
                    output_mode="student",
                    distill_bc=False,
                    distill_eta=False,
                )
            elif interv_encoder_type == "v1_trivalue_subgraph1hop":
                # Build the one-hop candidate subgraph once.
                sub_edge_index, ptb_to_node_sub, sub_nodes_full, node_map, cand_nodes_sub = build_candidate_1hop_subgraph(
                    base_edge_index=interv_edge_index,
                    ptb_to_node_full=ptb_to_node,
                    candidate_nodes_full=None,
                )
                sub_num_nodes = int(sub_nodes_full.numel())

                self.graph_c_encoder = FixedTriValueSmallGraphEncoder(
                    num_nodes=sub_num_nodes,
                    z_dim=self.z_dim,
                    edge_index=sub_edge_index,
                    ptb_to_node=ptb_to_node_sub,
                    candidate_nodes=cand_nodes_sub,
                    node_emb_dim=node_emb_dim,
                    gnn_hidden=gnn_hidden,
                    mlp_hidden=128,
                    readout=readout,
                    gnn_dropout=0.1,
                    use_ptb_strength=True,
                )
            else:
                raise ValueError(f"Unknown interv_encoder_type: {interv_encoder_type}")
            

        # Encoder for gene
        hids = 128
        self.fc1 = nn.Linear(self.dim, hids)
        weights_init(self.fc1)
        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)

        # DAG matrix G (upper triangular)
        self.G = torch.nn.Parameter(torch.normal(0, .1, size=(self.z_dim, self.z_dim)))

        # fallback C encoder (MLP)
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        # IMPORTANT: per-ptb baseline strength should be c_dim
        self.c_shift = nn.Parameter(torch.ones(self.c_dim))

        # GNN layer (your original gene-expression GNN)
        if mode in [5]:
            self.gnn_conv1 = GCNConv(1, 1)
        if mode in [3, 4, 6, 11]:
            self.gnn_conv1 = GATConv(1, 1)
        if mode in [13]:
            self.gnn_conv1 = GATConv(1, 4, 4)
            self.gnn_conv2 = GATConv(16, 8, 4)
            self.gnn_conv3 = GATConv(32, 1, 1)
        if mode in [15]:
            self.gnn_conv1 = SAGEConv(1, 1)

        # Decoder for latent state
        self.d1 = nn.Linear(self.z_dim, hids)
        self.d2 = nn.Linear(hids, self.dim)
        weights_init(self.d1)
        weights_init(self.d2)

        # activation
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        # expose last aux loss (e.g., distillation)
        self.c_encoder_aux_loss = None

    def encode_g(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h))

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return eps.mul(std).add_(mu)

    def decode(self, u):
        h = self.leakyrelu(self.d1(u))
        return self.leakyrelu(self.d2(h))

    def c_encode(self, c, temp=1.0):
        if self.graph_c_encoder is not None:
            bc, eta = self.graph_c_encoder(c, temp=temp)
            # store aux loss if encoder provides it
            aux = getattr(self.graph_c_encoder, "distill_loss", None)
            self.c_encoder_aux_loss = aux
            return bc, eta

        # fallback
        h = self.leakyrelu(self.c1(c))
        bc = self.sftmx(self.c2(h) * temp)
        eta = c @ self.c_shift
        self.c_encoder_aux_loss = None
        return bc, eta

    def dag(self, z, bc, csz, bc2, csz2, num_interv=1):
        I = torch.eye(self.z_dim).to(self.device)
        inv_mat = torch.inverse(I - torch.triu(self.G, diagonal=1))
        if num_interv == 0:
            return z @ inv_mat
        if num_interv == 1:
            zinterv = (z + bc * csz.reshape(-1, 1))
        else:
            zinterv = z + bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1)
        return zinterv @ inv_mat

    def forward(self, x, onehot, c, c2, edge_index, mode, num_interv=1, temp=1):
        assert num_interv in [0, 1, 2]
        bc, csz = self.c_encode(c, temp)
        if num_interv == 2:
            bc2, csz2 = self.c_encode(c2, temp)
        else:
            bc2, csz2 = bc, csz

        # pathway embedding branch (kept as your original logic)
        mu_p = torch.zeros(self.p_dim, device=self.device)
        var_p = torch.ones(self.p_dim, device=self.device)
        if self.p_dim > 0:
            z2 = self.reparametrize(mu_p, var_p).repeat(len(x), 1)
        if self.p_dim > 0:
            x_cat = torch.cat([x, z2], dim=1)
        else:
            x_cat = x
        x_cat = x_cat.unsqueeze(-1)

        # gene-expression GNN
        if mode in [3, 4, 11, 13, 6]:
            x_list = list(x_cat)
            x2 = [Data(x=i, edge_index=edge_index) for i in x_list]
            x2 = Batch.from_data_list(x2)
            gnn_out = self.gnn_conv1(x2.x, x2.edge_index)
            gnn_out = gnn_out.view(len(x_list), -1, gnn_out.shape[1])
            if mode in [13]:
                gnn_out = self.relu(gnn_out)
                gnn_list = list(gnn_out)
                x2 = [Data(x=i, edge_index=edge_index) for i in gnn_list]
                x2 = Batch.from_data_list(x2)
                gnn_out = self.gnn_conv2(x2.x, x2.edge_index)
                gnn_out = gnn_out.view(len(gnn_list), -1, gnn_out.shape[1])
                gnn_out = self.relu(gnn_out)
                gnn_list = list(gnn_out)
                x2 = [Data(x=i, edge_index=edge_index) for i in gnn_list]
                x2 = Batch.from_data_list(x2)
                gnn_out = self.gnn_conv3(x2.x, x2.edge_index)
                gnn_out = gnn_out.view(len(gnn_list), -1, gnn_out.shape[1])
        else:
            # modes [5,15]
            gnn_out = self.gnn_conv1(x_cat, edge_index)

        gnn_out = gnn_out.squeeze(-1)
        if mode in [3, 4, 5, 6, 11, 13, 15]:
            gnn_out = gnn_out[:, : self.dim]

        mu, var = self.encode_g(gnn_out)
        z = self.reparametrize(mu, var)
        u = self.dag(z, bc, csz, bc2, csz2, num_interv)
        y_hat = self.decode(u)

        # observational reconstruction
        u_recon = self.dag(z, bc * 0, csz * 0, bc * 0, csz * 0, num_interv=0)
        x_recon = self.decode(u_recon)

        return y_hat, x_recon, mu, var, self.G,bc,csz,bc2,csz2



class CMVAE(nn.Module):

    def __init__(self, dim, z_dim, c_dim, device=None):

        super(CMVAE, self).__init__()

        if device is None:
            self.cuda = False
            self.device = 'cpu'
        else:
            self.device = device
            self.cuda = True
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim
        self.p_dim=2694

        # encoder
        hids = 128
        self.fc1 = nn.Linear(self.dim,hids)

        weights_init(self.fc1)
        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)
        
        # DAG matrix G (upper triangular, z_dim x z_dim). 
        # encoded as a dense matrix, where only upper triangular parts will be used
        self.G = torch.nn.Parameter(torch.normal(0,.1,size = (self.z_dim,self.z_dim)))
        
        # C encoder
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        self.c_shift = nn.Parameter(torch.ones(self.c_dim))

        # decoder
        self.d1 = nn.Linear(self.z_dim,hids)
        self.d2 = nn.Linear(hids, self.dim)
        weights_init(self.d1)
        weights_init(self.d2)
        
        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h)) 

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.DoubleTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.DoubleTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def decode(self, u):        
        h = self.leakyrelu(self.d1(u))
        return self.leakyrelu(self.d2(h))
    
    def c_encode(self, c, temp=1):
        h = self.leakyrelu(self.c1(c))
        h = self.sftmx(self.c2(h)*temp)
        s = c @ self.c_shift
        return h, s
    
    # Causal DAG "layer"
    # bc is a softmax vector encoding the target of the intervetnion
    # csz encodes the strength of the intervention
    def dag(self, z, bc, csz, bc2, csz2, num_interv = 1):
        if num_interv == 0:
            u = (z) @ torch.inverse(torch.eye(self.z_dim).to(self.device) - torch.triu((self.G), diagonal=1))
        else:
            if num_interv == 1: # 1 - bc
                zinterv = z * (1.) + bc * csz.reshape(-1,1)
            else: # 1. - bc - bc2
                zinterv = z * (1.) + bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1)
            
            u = (zinterv) @ torch.inverse(torch.eye(self.z_dim).to(self.device) -  torch.triu((self.G), diagonal=1))     
        return u

    def forward(self, x, c, c2, num_interv = 1, temp = 1):
        assert num_interv in [0,1,2], "support single- or double-node interventions only"

        # decode an interventional sample from an observational sample    
        bc, csz = self.c_encode(c, temp)       
        bc2, csz2 = self.c_encode(c2, temp)
        #mu_p= torch.zeros(self.p_dim,device=self.device)
        
        #var_p=torch.ones(self.p_dim,device=self.device)

        mu, var = self.encode(x)
        z = self.reparametrize(mu, var)
        u = self.dag(z, bc, csz, bc2, csz2, num_interv)
    
        y_hat = self.decode(u)
        
        # create the reconstruction of observational sample
        u_recon = self.dag(z, bc*0, csz*0, bc*0, csz*0, num_interv=0)
        x_recon = self.decode(u_recon)
        
        return y_hat, x_recon, mu, var, self.G


# Baseline models
# conditional VAE with causal layer but no mmd loss
class CGVAE(nn.Module):
    def __init__(self, dim, z_dim, c_dim, p_dim, mode, device=None):
        super(CGVAE, self).__init__()
        
        self.device = 'cpu' if device == 'cpu' else device
        self.cuda = (self.device != 'cpu')
        
        self.dim = dim
        self.z_dim = z_dim
        self.p_dim = p_dim
        self.c_dim = c_dim
        self.mode = mode
        # C encoder
        hids = 128
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        self.c_shift = nn.Parameter(torch.ones(self.z_dim))
   

        self.fc1 = nn.Linear(self.dim, hids)
        weights_init(self.fc1)

        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)

        # Additive shift module: maps c to z_dim
        self.c_shift_mlp = nn.Sequential(
            nn.Linear(c_dim, hids),
            nn.LeakyReLU(0.2),
            nn.Linear(hids, z_dim)
        )

        # Decoder
        self.d1 = nn.Linear(z_dim, hids)
        self.d2 = nn.Linear(hids, dim)
        weights_init(self.d1)
        weights_init(self.d2)

        # GNN Encoder (if applicable)

        self.gnn_conv1 = SAGEConv(1, 1)

                # Activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return eps.mul(std).add_(mu)
    def c_encode(self, c, temp=1):
        h = self.leakyrelu(self.c1(c))
        h = self.sftmx(self.c2(h) * temp)
        s = c @ self.c_shift
        return h, s
    def encode(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h))

    def decode(self, z):
        h = self.leakyrelu(self.d1(z))
        return self.leakyrelu(self.d2(h))
    def latent(self, z, bc, csz, bc2, csz2, num_interv = 1):
        if num_interv == 0:
            u = z
        else:
            if num_interv == 1:
                zinterv = z * (1.) + bc * csz.reshape(-1,1)
            else:
                zinterv = z * (1.) + bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1)
            
            u = zinterv   
        return u
    def  forward(self, x, onehot,c, c2, edge_index, mode, num_interv=1, temp=1):
        # Concatenate pathway embedding if needed
        bc, csz = self.c_encode(c, temp)       
        bc2, csz2 = self.c_encode(c2, temp)
        mu_p= torch.zeros(self.p_dim,device=self.device)
        
        var_p=torch.ones(self.p_dim,device=self.device)
        if self.p_dim>0:
           z2 = self.reparametrize(mu_p, var_p)
           z2=z2.repeat(len(x),1)
        if self.p_dim > 0:
            x_cat = torch.cat([x, z2], dim=1)
        else:
            x_cat = x
        x_cat = x_cat.unsqueeze(-1)


        gnn_out = self.gnn_conv1(x_cat, edge_index)

        gnn_out = gnn_out.squeeze(-1)

        gnn_out = gnn_out[:, :self.dim]

        mu, var = self.encode(gnn_out)
        z = self.reparametrize(mu, var)
        u = self.latent(z, bc, csz, bc2, csz2, num_interv)
    
        y_hat = self.decode(u)
        
        # create the reconstruction of observational sample
        u_recon = self.latent(z, bc*0, csz*0, bc*0, csz*0, num_interv=0)
        x_recon = self.decode(u_recon)
        
        return  y_hat, x_recon, mu, var, None


class CMVAEGNN(nn.Module):
    def __init__(
        self,
        dim,
        z_dim,
        c_dim,
        p_dim,
        mode,
        device=None,
        use_external_embedding=False,
        external_embedding_dim=0,
    ):#p_dim=2082(pathways involved with at least one 5000 genes)
        super(CMVAEGNN, self).__init__()
        if mode not in SUPPORTED_GNN_MODES:
            raise ValueError(f"Unsupported GNN mode: {mode}")
        
        if device == 'cpu':
            self.cuda = False
            self.device = 'cpu'
        
        else:
            self.device = device
            self.cuda = True
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim
        self.p_dim = p_dim
        self.use_external_embedding = bool(use_external_embedding)
        self.external_embedding_dim = int(external_embedding_dim)
        
        # Encoder for gene
        hids = 128
        if self.use_external_embedding:
           self.fc1 = nn.Linear(self.dim + self.external_embedding_dim, hids)
        else:
            self.fc1 = nn.Linear(self.dim, hids)
        weights_init(self.fc1)
        
        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)
        # DAG matrix G (upper triangular, z_dim x z_dim)
        self.G = torch.nn.Parameter(torch.normal(0, .1, size=(self.z_dim, self.z_dim)))
        
        # C encoder
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        self.c_shift = nn.Parameter(torch.ones(self.z_dim))

        # GNN Layer
        if mode in [5]:
            self.gnn_conv1 = GCNConv(1, 1)
        
        if mode in [3,4,6,11]:
            self.gnn_conv1 = GATConv(1, 1)# GATConv(1, 16)
        if mode in [13]:
            self.gnn_conv1 = GATConv(1, 4,4)
            self.gnn_conv2 = GATConv(16, 8,4)## GATConv(16, 8)
            self.gnn_conv3 = GATConv(32, 1,1)# GATConv(8, 1)

        if mode in [15]:
            self.gnn_conv1=SAGEConv(1,1)

        # Decoder
        self.d1 = nn.Linear(self.z_dim, hids)#from p_dim to z_dim
        self.d2 = nn.Linear(hids, self.dim)
        weights_init(self.d1)
        weights_init(self.d2)
        
        # Activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode_g(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h)) 
    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return eps.mul(std).add_(mu) 

    def decode(self, u):
        h = self.leakyrelu(self.d1(u))
        return self.leakyrelu(self.d2(h))
    
    def c_encode(self, c, temp=1):
        h = self.leakyrelu(self.c1(c))
        h = self.sftmx(self.c2(h) * temp)
        s = c @ self.c_shift
        return h, s
    
    def dag(self, z, bc, csz, bc2, csz2, num_interv=1):
        if num_interv == 0:
            u = (z) @ torch.inverse(torch.eye(self.z_dim).to(self.device) - torch.triu(self.G, diagonal=1))# from p_dim to z_dim
        else:
            if num_interv == 1:
                zinterv = z * 1. + bc * csz.reshape(-1, 1)
            elif num_interv == 2:
                zinterv = z * 1. + bc * csz.reshape(-1, 1) + bc2 * csz2.reshape(-1, 1)
            u = (zinterv) @ torch.inverse(torch.eye(self.z_dim).to(self.device) - torch.triu(self.G, diagonal=1))
        return u

    def forward(self, x, onehot,c, c2, edge_index, mode, num_interv=1, temp=1):
        assert num_interv in [0, 1, 2], "support single- or double-node interventions only"

        # Decode an interventional sample from an observational sample    
        bc, csz = self.c_encode(c, temp)       
        bc2, csz2 = self.c_encode(c2, temp)
        
        # Optional extra embedding branch.
        extra_embedding = None
        if self.use_external_embedding:
            if isinstance(onehot, torch.Tensor):
                extra_embedding = onehot
            else:
                raise RuntimeError("CMVAEGNN expected an external embedding tensor but did not receive one.")
            # Keep the original graph node count for gene+pathway graphs by padding
            # the pathway part with zeros, then concatenate the external cell embedding
            # only after the GNN has produced gene-level outputs.
            if self.p_dim > 0:
                z2 = torch.zeros((len(x), self.p_dim), device=x.device, dtype=x.dtype)
                x_cat = torch.cat([x, z2], dim=1)
            else:
                x_cat = x
        else:
            mu_p= torch.zeros(self.p_dim,device=self.device)
            var_p=torch.ones(self.p_dim,device=self.device)
            if self.p_dim>0:
               z2 = self.reparametrize(mu_p, var_p)
               z2=z2.repeat(len(x),1)

            if self.p_dim>0:
               x_cat = torch.cat([x,z2], dim=1)
            else:
                x_cat=x
        x_cat = x_cat.unsqueeze(-1)

        if mode in [3,4,11,13,6]:
            x_cat = list(x_cat) 
            x2 = [Data(x=i, edge_index=edge_index) for i in x_cat] 
            x2 = Batch.from_data_list(x2)
            gnn_out=self.gnn_conv1(x2.x,x2.edge_index)
            gnn_out=gnn_out.view(len(x_cat),-1,gnn_out.shape[1])
            
            if mode in [13]:
                gnn_out=self.relu(gnn_out)
                gnn_out = list(gnn_out) 
                x2 = [Data(x=i, edge_index=edge_index) for i in gnn_out] 
                x2 = Batch.from_data_list(x2)
                gnn_out=self.gnn_conv2(x2.x,x2.edge_index)
                gnn_out=gnn_out.view(len(x_cat),-1,gnn_out.shape[1])
                gnn_out=self.relu(gnn_out)
                gnn_out = list(gnn_out) 
                x2 = [Data(x=i, edge_index=edge_index) for i in gnn_out] 
                x2 = Batch.from_data_list(x2)
                gnn_out=self.gnn_conv3(x2.x,x2.edge_index)
                gnn_out=gnn_out.view(len(x_cat),-1,gnn_out.shape[1])
        # Apply GNN
        if mode in [5,15]:
            gnn_out = self.gnn_conv1(x_cat, edge_index)

        gnn_out = gnn_out.squeeze(-1)#[:,:self.dim] #find embeddings for genes
        if mode in [3,4,5,6,11,13,15]:
             gnn_out= gnn_out[:,:self.dim]
        if self.use_external_embedding:
            gnn_out = torch.cat([gnn_out, extra_embedding], dim=1)
        mu, var = self.encode_g(gnn_out)
        z = self.reparametrize(mu, var)
        # Apply DAG layer
        u = self.dag(z, bc, csz, bc2, csz2, num_interv)
    
        y_hat = self.decode(u)
        
        # Create the reconstruction of observational sample
        u_recon = self.dag(z, bc * 0, csz * 0, bc * 0, csz * 0, num_interv=0)
        x_recon = self.decode(u_recon)
        
        return y_hat, x_recon, mu, var, self.G
    


class CMVAEonehot(nn.Module):
    def __init__(self, dim, z_dim, c_dim, p_dim,device=None):#p_dim=2082(pathways involved with at least one 5000 genes)
        super(CMVAEonehot, self).__init__()
        
        if device == 'cpu':
            self.cuda = False
            self.device = 'cpu'
        
        else:
            self.device = device
            self.cuda = True
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim
        self.p_dim = p_dim
        # Encoder for gene
        hids = 128
        self.fc1 = nn.Linear(self.dim+self.p_dim, hids)
        weights_init(self.fc1)
        
        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)
        


    
        # DAG matrix G (upper triangular, z_dim x z_dim)
        self.G = torch.nn.Parameter(torch.normal(0, .1, size=(self.z_dim, self.z_dim)))
        
        # C encoder
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        self.c_shift = nn.Parameter(torch.ones(self.z_dim))


        # Decoder
        self.d1 = nn.Linear(self.z_dim, hids)#from p_dim to z_dim
        self.d2 = nn.Linear(hids, self.dim)
        weights_init(self.d1)
        weights_init(self.d2)
        
        # Activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode_g(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h)) 


    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return eps.mul(std).add_(mu) 

    def decode(self, u):
        h = self.leakyrelu(self.d1(u))
        return self.leakyrelu(self.d2(h))
    
    def c_encode(self, c, temp=1):
        h = self.leakyrelu(self.c1(c))
        h = self.sftmx(self.c2(h) * temp)
        s = c @ self.c_shift
        return h, s
    
    def dag(self, z, bc, csz, bc2, csz2, num_interv=1):
        if num_interv == 0:
            u = (z) @ torch.inverse(torch.eye(self.z_dim).to(self.device) - torch.triu(self.G, diagonal=1))# from p_dim to z_dim
        else:
            if num_interv == 1:
                zinterv = z * 1. + bc * csz.reshape(-1, 1)
            elif num_interv == 2:
                zinterv = z * 1. + bc * csz.reshape(-1, 1) + bc2 * csz2.reshape(-1, 1)
            
            u = (zinterv) @ torch.inverse(torch.eye(self.z_dim).to(self.device) - torch.triu(self.G, diagonal=1))
        return u

    def forward(self, x,onehot, c, c2, mode, num_interv=1, temp=1):
        assert num_interv in [0, 1, 2], "support single- or double-node interventions only"
        # Decode an interventional sample from an observational sample    
        bc, csz = self.c_encode(c, temp)       
        bc2, csz2 = self.c_encode(c2, temp)
        x_concat=torch.cat((x,onehot),dim=1)
        mu, var = self.encode_g(x_concat)
        z = self.reparametrize(mu, var)
        u = self.dag(z, bc, csz, bc2, csz2, num_interv)
    
        y_hat = self.decode(u)
        
        # create the reconstruction of observational sample
        u_recon = self.dag(z, bc*0, csz*0, bc*0, csz*0, num_interv=0)
        x_recon = self.decode(u_recon)
        
        return y_hat, x_recon, mu, var, self.G
