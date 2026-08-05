import scanpy as sc
import numpy as np
import torch
import pandas as pd
import random

def read_gmt(file_path):
    gmt_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            gene_set_name = parts[0]
            gene_set_description = parts[1]
            genes = parts[2:]
            gmt_dict[gene_set_name] = {
                "description": gene_set_description,
                "genes": genes
            }
    return gmt_dict

import torch

def build_gene_union_graph(device, genes_X, ptb_targets):
    """
    genes_X: 5000 HVG 基因列表（顺序要与 adata.var.gene_symbols 一致）
    ptb_targets: 105 intervention gene 列表（顺序要与 c 的 one-hot 维度一致）

    返回:
      genes_all: union 后的基因节点列表
      edge_index: gene-gene 图（只含基因节点，不含pathway） [2, E]
      ptb_to_node: [c_dim]，第 i 个 intervention gene 对应的 node id
    """
    genes_all = list(genes_X)
    for g in ptb_targets:
        if g not in genes_all:
            genes_all.append(g)

    edge_index = getedge_index_GG(device, genes_all)

    gene_to_idx = {g: i for i, g in enumerate(genes_all)}
    ptb_to_node = torch.tensor([gene_to_idx[g] for g in ptb_targets],
                               dtype=torch.long, device=device)
    return genes_all, edge_index, ptb_to_node


def build_pathway_dag_components(device, genes_A, ptb_targets=None, p_dim=None):
    gmt_file_path = './ReactomePathways.gmt'
    gmt_data = read_gmt(gmt_file_path)

    df_gene_pathway = pd.DataFrame(
        [(value["description"], gene) for value in gmt_data.values() for gene in value["genes"]],
        columns=["Pathway", "Gene"]
    )

    raw_pathway_names = df_gene_pathway["Pathway"].drop_duplicates().tolist()
    file_path = './ReactomePathwaysRelation.txt'
    df_pathway_relationship = pd.read_csv(file_path, sep='\t', header=None, names=['Pathway1', 'Pathway2'])

    adj = {name: [] for name in raw_pathway_names}
    indegree = {name: 0 for name in raw_pathway_names}
    for _, row in df_pathway_relationship.iterrows():
        src = row['Pathway1']
        dst = row['Pathway2']
        if src in adj and dst in adj and src != dst:
            adj[src].append(dst)
            indegree[dst] += 1

    queue = sorted([name for name, deg in indegree.items() if deg == 0])
    ordered = []
    while queue:
        cur = queue.pop(0)
        ordered.append(cur)
        for nxt in sorted(adj[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
        queue.sort()

    if len(ordered) < len(raw_pathway_names):
        remaining = sorted([name for name in raw_pathway_names if name not in set(ordered)])
        ordered.extend(remaining)

    pathway_names = ordered
    if p_dim is not None and p_dim > 0:
        pathway_names = pathway_names[:min(int(p_dim), len(pathway_names))]
    pathway_to_index = {pathway: idx for idx, pathway in enumerate(pathway_names)}

    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)}
    gene_pathway = torch.zeros((len(genes_A), len(pathway_names)), dtype=torch.double)

    for _, row in df_gene_pathway.iterrows():
        pathway = row["Pathway"]
        gene = row["Gene"]
        if pathway in pathway_to_index and gene in gene_to_index:
            gene_pathway[gene_to_index[gene], pathway_to_index[pathway]] = 1.0

    if gene_pathway.numel() > 0:
        pathway_gene_count = gene_pathway.sum(dim=0, keepdim=True).clamp_min(1.0)
        gene_pathway = gene_pathway / pathway_gene_count

    ptb_to_pathway = None
    if ptb_targets is not None:
        ptb_to_pathway = torch.zeros((len(ptb_targets), len(pathway_names)), dtype=torch.double)
        gene_rows = df_gene_pathway.groupby("Gene")["Pathway"].apply(list).to_dict()
        for i, gene in enumerate(ptb_targets):
            for pathway in gene_rows.get(gene, []):
                if pathway in pathway_to_index:
                    ptb_to_pathway[i, pathway_to_index[pathway]] = 1.0
        if ptb_to_pathway.numel() > 0:
            row_sums = ptb_to_pathway.sum(dim=1, keepdim=True).clamp_min(1.0)
            ptb_to_pathway = ptb_to_pathway / row_sums

    adj_matrix = np.zeros((len(pathway_names), len(pathway_names)), dtype=int)
    for _, row in df_pathway_relationship.iterrows():
        pathway1, pathway2 = row['Pathway1'], row['Pathway2']
        if pathway1 in pathway_to_index and pathway2 in pathway_to_index and pathway1 != pathway2:
            idx1 = pathway_to_index[pathway1]
            idx2 = pathway_to_index[pathway2]
            if idx1 < idx2:
                adj_matrix[idx1, idx2] = 1
            elif idx2 < idx1:
                adj_matrix[idx2, idx1] = 1

    pathway_edge_index = np.array(np.nonzero(adj_matrix))
    pathway_edge_index = torch.tensor(pathway_edge_index, dtype=torch.long, device=device)

    gene_pathway = gene_pathway.to(device)
    if ptb_to_pathway is not None:
        ptb_to_pathway = ptb_to_pathway.to(device)

    return pathway_names, gene_pathway, pathway_edge_index, ptb_to_pathway

def getedge_index_GG(device,genes_A):


# Create a mapping of genes to index
    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)} 


# Initialize the edge index list
    edge_index = []


# 邻接矩阵的维度
    matrix_size = len(genes_A) 
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    file_path_g='./genes_edgelist.txt'
    df_gene_relationship=pd.read_csv(file_path_g, sep=' ', header=None,names=['Gene1', 'Gene2'])
    for _, row in df_gene_relationship.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']

        if gene1 in gene_to_index and gene2 in gene_to_index and gene1!=gene2:
            gene1_idx = gene_to_index[gene1]
            gene2_idx = gene_to_index[gene2]
            adj_matrix[gene1_idx, gene2_idx] = 1
            adj_matrix[gene2_idx, gene1_idx] = 1  

    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)



    return edge_index
def build_candidate_1hop_subgraph(
    base_edge_index: torch.Tensor,     # [2, E] on FULL graph node ids (0..N_full-1)
    ptb_to_node_full: torch.Tensor,    # [c_dim] each ptb -> global node id
    candidate_nodes_full: torch.Tensor = None,  # optional [105] global ids; default ptb_to_node_full.unique()
):
    """
    Returns:
      sub_edge_index: [2, E_sub] with local node ids (0..N_sub-1)
      ptb_to_node_sub: [c_dim] mapping ptb -> local node id
      sub_nodes_full: [N_sub] list of global node ids included (for debugging/optional)
      node_map_full_to_sub: dict-like tensor [N_full] -> local id or -1
      candidate_nodes_sub: [<=105] local ids
    """
    ei = base_edge_index.long()
    src, dst = ei[0], ei[1]

    if candidate_nodes_full is None:
        cand_full = ptb_to_node_full.unique().long()
    else:
        cand_full = candidate_nodes_full.long()

    # incident edges to candidates
    if hasattr(torch, "isin"):
        incident = torch.isin(src, cand_full) | torch.isin(dst, cand_full)
    else:
        # fallback slow, but cand_full is small
        cand_set = set(cand_full.detach().cpu().tolist())
        incident = torch.tensor([(int(s.item() in cand_set) or int(d.item() in cand_set))
                                 for s, d in zip(src, dst)], device=src.device, dtype=torch.bool)

    neigh_full = torch.cat([src[incident], dst[incident]], dim=0).unique()
    sub_nodes_full = torch.cat([cand_full, neigh_full], dim=0).unique()   # [N_sub] global ids

    # build mapping full -> sub local
    N_full = int(max(int(src.max()), int(dst.max()), int(ptb_to_node_full.max())) + 1)
    node_map = torch.full((N_full,), -1, dtype=torch.long, device=ei.device)
    node_map[sub_nodes_full] = torch.arange(sub_nodes_full.numel(), device=ei.device)

    # induced edges among sub_nodes
    in_sub_src = node_map[src] >= 0
    in_sub_dst = node_map[dst] >= 0
    mask = in_sub_src & in_sub_dst
    edge_sub_full = ei[:, mask]
    sub_edge_index = node_map[edge_sub_full]   # now local ids

    # ptb_to_node mapping into subgraph local ids
    ptb_to_node_sub = node_map[ptb_to_node_full.long()]  # [c_dim]
    # sanity: all ptb nodes must be in subgraph
    if (ptb_to_node_sub < 0).any():
        bad = torch.where(ptb_to_node_sub < 0)[0][:10].detach().cpu().tolist()
        raise ValueError(f"Some ptb nodes not in subgraph. First bad ptb indices: {bad}")

    candidate_nodes_sub = node_map[cand_full]
    return sub_edge_index, ptb_to_node_sub, sub_nodes_full, node_map, candidate_nodes_sub

def allconnected_index_GG(device, genes_A):
    # Create a mapping of genes to index
    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)} 

    # 获取基因数量
    matrix_size = len(genes_A) 

    # 构建全连接图的 edge_index，不允许自连接
    row = torch.arange(matrix_size).repeat(matrix_size)  # 行: 每个节点的索引
    col = torch.arange(matrix_size).repeat_interleave(matrix_size)  # 列: 每个节点的索引

    # 创建全连接图后去除自连接
    mask = row != col  # 排除 self-loops (即 row == col)
    edge_index = torch.stack([row[mask], col[mask]], dim=0)  # 只保留非自连接的边

    # 为确保图是无向图，我们需要添加每条边的反向边
    edge_index_reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)  # 反转每条边

    # 合并原始边和反向边
    edge_index = torch.cat([edge_index, edge_index_reverse], dim=1)

    # 将 edge_index 转移到设备上
    edge_index = edge_index.to(device)

    return edge_index








def getedge_index_GP(device,genes_A):
    gmt_file_path = './ReactomePathways.gmt'
    gmt_data = read_gmt(gmt_file_path)


    df_B = pd.DataFrame([(key, value["description"], gene) for key, value in gmt_data.items() for gene in value["genes"]],columns=["Gene Set", "Description", "Gene"])

# Create a mapping of genes to index
    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)} #length 105

# Create a list of unique pathways
    involved_pathways = df_B[df_B['Gene'].isin(genes_A)]['Gene Set'].unique().tolist()
    

# Create a mapping of pathways to index
    pathway_to_index = {pathway: idx + len(genes_A) for idx, pathway in enumerate(involved_pathways)}

# Initialize the edge index list
    edge_index = []


# 邻接矩阵的维度
    matrix_size = len(genes_A) + len(involved_pathways)
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    #print(len(involved_pathways))

# 填充邻接矩阵
    for _, row in df_B.iterrows():
          gene, pathway = row['Gene'], row['Gene Set']
          if gene in gene_to_index and pathway in pathway_to_index:
             gene_idx = gene_to_index[gene]
             pathway_idx = pathway_to_index[pathway]
             adj_matrix[gene_idx, pathway_idx] = 1
             adj_matrix[pathway_idx, gene_idx] = 1  # 假设无向图

# 转换为 edge_index 格式
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

# 打印结果
    #print(f"Edge Index Shape: {edge_index.shape}")

    return edge_index



def getedge_index_GGGP(device,genes_A):
    gmt_file_path = './ReactomePathways.gmt'
    gmt_data = read_gmt(gmt_file_path)


    df_B = pd.DataFrame([(key, value["description"], gene) for key, value in gmt_data.items() for gene in value["genes"]],columns=["Gene Set", "Description", "Gene"])

# Create a mapping of genes to index
    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)} #length 105

# Create a list of unique pathways
    involved_pathways = df_B[df_B['Gene'].isin(genes_A)]['Description'].unique().tolist()
    

# Create a mapping of pathways to index
    pathway_to_index = {pathway: idx + len(genes_A) for idx, pathway in enumerate(involved_pathways)}

# Initialize the edge index list
    edge_index = []


# 邻接矩阵的维度
    matrix_size = len(genes_A) + len(involved_pathways)
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for _, row in df_B.iterrows():
          gene, pathway = row['Gene'], row['Description']
          if gene in gene_to_index and pathway in pathway_to_index:
             gene_idx = gene_to_index[gene]
             pathway_idx = pathway_to_index[pathway]
             adj_matrix[gene_idx, pathway_idx] = 1
             adj_matrix[pathway_idx, gene_idx] = 1  # 假设无向图
    file_path_g='./genes_edgelist.txt'
    df_gene_relationship=pd.read_csv(file_path_g, sep=' ', header=None,names=['Gene1', 'Gene2'])
    for _, row in df_gene_relationship.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']

        if gene1 in gene_to_index and gene2 in gene_to_index and gene1!=gene2:
            gene1_idx = gene_to_index[gene1]
            gene2_idx = gene_to_index[gene2]
            adj_matrix[gene1_idx, gene2_idx] = 1
            adj_matrix[gene2_idx, gene1_idx] = 1  

    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)



    return edge_index


def getedge_index_all(device,genes_A):
    gmt_file_path = './ReactomePathways.gmt'
    gmt_data = read_gmt(gmt_file_path)


    df_B = pd.DataFrame([(key, value["description"], gene) for key, value in gmt_data.items() for gene in value["genes"]],columns=["Gene Set", "Description", "Gene"])

# Create a mapping of genes to index
    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)} #length 105
    
# Create a list of unique pathways
    involved_pathways = df_B[df_B['Gene'].isin(genes_A)]['Description'].unique().tolist()
    

# Create a mapping of pathways to index
    pathway_to_index = {pathway: idx + len(genes_A) for idx, pathway in enumerate(involved_pathways)}

# Initialize the edge index list
    edge_index = []


# 邻接矩阵的维度
    matrix_size = len(genes_A) + len(involved_pathways)
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    #print(len(involved_pathways))

# 填充邻接矩阵edge
    for _, row in df_B.iterrows():
          gene, pathway = row['Gene'], row['Description']
          if gene in gene_to_index and pathway in pathway_to_index:
             gene_idx = gene_to_index[gene]
             pathway_idx = pathway_to_index[pathway]
             adj_matrix[gene_idx, pathway_idx] = 1
             adj_matrix[pathway_idx, gene_idx] = 1  
    file_path='./ReactomePathwaysRelation.txt'
    df_pathway_relationship=pd.read_csv(file_path, sep='\t', header=None,names=['Pathway1', 'Pathway2'])
    for _, row in df_pathway_relationship.iterrows():
        pathway1, pathway2 = row['Pathway1'], row['Pathway2']
        if pathway1 in pathway_to_index and pathway2 in pathway_to_index:
            
            pathway1_idx = pathway_to_index[pathway1]
            pathway2_idx = pathway_to_index[pathway2]
            adj_matrix[pathway1_idx, pathway2_idx] = 1
            adj_matrix[pathway2_idx, pathway1_idx] = 1 
    file_path_g='./genes_edgelist.txt'
    df_gene_relationship=pd.read_csv(file_path_g, sep=' ', header=None,names=['Gene1', 'Gene2'])
    for _, row in df_gene_relationship.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']

        if gene1 in gene_to_index and gene2 in gene_to_index and gene1!=gene2:
            
            gene1_idx = gene_to_index[gene1]
            gene2_idx = gene_to_index[gene2]
            adj_matrix[gene1_idx, gene2_idx] = 1
            adj_matrix[gene2_idx, gene1_idx] = 1  

# 转换为 edge_index 格式
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

# 打印结果
    #print(f"Edge Index Shape: {edge_index.shape}")

    return edge_index






def getedge_index_all_pathway(device,genes_A):
    gmt_file_path = './ReactomePathways.gmt'
    gmt_data = read_gmt(gmt_file_path)


    df_B = pd.DataFrame([(key, value["description"], gene) for key, value in gmt_data.items() for gene in value["genes"]],columns=["Gene Set", "Description", "Gene"])

# Create a mapping of genes to index
    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)} #length 105
    
# Create a list of unique pathways
    involved_pathways = df_B['Description'].unique().tolist()

# Create a mapping of pathways to index
    pathway_to_index = {pathway: idx + len(genes_A) for idx, pathway in enumerate(involved_pathways)}

# Initialize the edge index list
    edge_index = []


# 邻接矩阵的维度
    matrix_size = len(genes_A) + len(involved_pathways)
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    #print(len(involved_pathways))

# 填充邻接矩阵
    for _, row in df_B.iterrows():
          gene, pathway = row['Gene'], row['Description']
          if gene in gene_to_index and pathway in pathway_to_index:
             gene_idx = gene_to_index[gene]
             pathway_idx = pathway_to_index[pathway]
             adj_matrix[gene_idx, pathway_idx] = 1
             adj_matrix[pathway_idx, gene_idx] = 1  
    file_path='./ReactomePathwaysRelation.txt'
    df_pathway_relationship=pd.read_csv(file_path, sep='\t', header=None,names=['Pathway1', 'Pathway2'])
    for _, row in df_pathway_relationship.iterrows():
        pathway1, pathway2 = row['Pathway1'], row['Pathway2']
        if pathway1 in pathway_to_index and pathway2 in pathway_to_index:
            
            pathway1_idx = pathway_to_index[pathway1]
            pathway2_idx = pathway_to_index[pathway2]
            adj_matrix[pathway1_idx, pathway2_idx] = 1
            adj_matrix[pathway2_idx, pathway1_idx] = 1 
    file_path_g='./genes_edgelist.txt'
    df_gene_relationship=pd.read_csv(file_path_g, sep=' ', header=None,names=['Gene1', 'Gene2'])
    for _, row in df_gene_relationship.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']

        if gene1 in gene_to_index and gene2 in gene_to_index and gene1!=gene2:
            
            gene1_idx = gene_to_index[gene1]
            gene2_idx = gene_to_index[gene2]
            adj_matrix[gene1_idx, gene2_idx] = 1
            adj_matrix[gene2_idx, gene1_idx] = 1  

# 转换为 edge_index 格式
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

# 打印结果
    #print(f"Edge Index Shape: {edge_index.shape}")

    return edge_index




def random_index_all_pathway(device, genes_A,seed=42):
    random.seed(seed)
    np.random.seed(seed)
    original_edge_index = getedge_index_all_pathway(device, genes_A).cpu().numpy()
    num_nodes = int(original_edge_index.max()) + 1

    # Treat the graph as undirected and preserve the number of unique undirected edges.
    undirected_edges = set()
    for src, dst in zip(original_edge_index[0], original_edge_index[1]):
        if src == dst:
            continue
        a, b = sorted((int(src), int(dst)))
        undirected_edges.add((a, b))

    num_undirected_edges = len(undirected_edges)
    max_possible = num_nodes * (num_nodes - 1) // 2
    if num_undirected_edges > max_possible:
        raise ValueError("Number of edges exceeds number of possible undirected edges.")

    chosen = set()
    while len(chosen) < num_undirected_edges:
        a = random.randrange(num_nodes)
        b = random.randrange(num_nodes)
        if a == b:
            continue
        edge = tuple(sorted((a, b)))
        chosen.add(edge)

    directed_edges = []
    for a, b in chosen:
        directed_edges.append((a, b))
        directed_edges.append((b, a))

    edge_index = np.array(directed_edges, dtype=np.int64).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    return edge_index


def half_preserved_random_index_all_pathway(device, genes_A, seed=42, preserve_ratio=0.5):
    random.seed(seed)
    np.random.seed(seed)
    original_edge_index = getedge_index_all_pathway(device, genes_A).cpu().numpy()
    num_nodes = int(original_edge_index.max()) + 1

    undirected_edges = set()
    for src, dst in zip(original_edge_index[0], original_edge_index[1]):
        if src == dst:
            continue
        undirected_edges.add(tuple(sorted((int(src), int(dst)))))

    undirected_edges = sorted(undirected_edges)
    num_total_edges = len(undirected_edges)
    num_preserve = int(round(num_total_edges * float(preserve_ratio)))
    num_preserve = max(0, min(num_total_edges, num_preserve))

    preserved = set(random.sample(undirected_edges, num_preserve))
    chosen = set(preserved)

    while len(chosen) < num_total_edges:
        a = random.randrange(num_nodes)
        b = random.randrange(num_nodes)
        if a == b:
            continue
        edge = tuple(sorted((a, b)))
        if edge in chosen:
            continue
        chosen.add(edge)

    directed_edges = []
    for a, b in sorted(chosen):
        directed_edges.append((a, b))
        directed_edges.append((b, a))

    edge_index = np.array(directed_edges, dtype=np.int64).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    return edge_index


def mixed_half_correct_half_random_index_all_pathway(device, genes_A, seed=42):
    """
    Build one graph with the same total edge count as the true all-pathway graph:
    - 50% of undirected edges are kept from the true graph
    - 50% of undirected edges are newly random rewired edges

    Returned edge_index is bidirectional, matching the rest of the codebase.
    """
    return half_preserved_random_index_all_pathway(
        device=device,
        genes_A=genes_A,
        seed=seed,
        preserve_ratio=0.5,
    )



'''
adata = sc.read_h5ad('./cpa_binaries/datasets/Norman2019_raw.h5ad')
genes_A=list(adata.var.gene_symbols)
ptb_targets = list(set().union(*[set(i.split(',')) for i in adata.obs['guide_ids'].value_counts().index]))
ptb_targets.remove('')
genes,egde,node=build_gene_union_graph('cuda:0',genes_A,ptb_targets)
#edge=getedge_index_all("cuda:0",genes_A)
print(len(genes))
print(egde.shape)
print(node.shape)
'''
