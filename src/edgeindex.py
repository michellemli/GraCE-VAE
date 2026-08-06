import numpy as np
import torch
import pandas as pd
import random
import project_config

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

def build_gene_union_graph(device, genes_X, ptb_targets):
    """
    genes_X: 5,000 HVGs in the same order as adata.var.gene_symbols.
    ptb_targets: 105 intervention genes in the same order as the one-hot c.

    Returns:
      genes_all: union of observed and intervention genes.
      edge_index: gene-gene graph without pathway nodes, shape [2, E].
      ptb_to_node: node id for each intervention gene, shape [c_dim].
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


def getedge_index_GG(device,genes_A):


# Create a mapping of genes to index
    gene_to_index = {gene: idx for idx, gene in enumerate(genes_A)} 


# Initialize the edge index list
    edge_index = []


# Adjacency matrix dimension.
    matrix_size = len(genes_A) 
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    file_path_g=project_config.NETWORK_DIR / 'genes_edgelist.txt'
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

def getedge_index_GGGP(device,genes_A):
    gmt_file_path = project_config.NETWORK_DIR / 'ReactomePathways.gmt'
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


# Adjacency matrix dimension.
    matrix_size = len(genes_A) + len(involved_pathways)
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for _, row in df_B.iterrows():
          gene, pathway = row['Gene'], row['Description']
          if gene in gene_to_index and pathway in pathway_to_index:
             gene_idx = gene_to_index[gene]
             pathway_idx = pathway_to_index[pathway]
             adj_matrix[gene_idx, pathway_idx] = 1
             adj_matrix[pathway_idx, gene_idx] = 1  # Undirected graph.
    file_path_g=project_config.NETWORK_DIR / 'genes_edgelist.txt'
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
    gmt_file_path = project_config.NETWORK_DIR / 'ReactomePathways.gmt'
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


# Adjacency matrix dimension.
    matrix_size = len(genes_A) + len(involved_pathways)
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    #print(len(involved_pathways))

# Populate the adjacency matrix.
    for _, row in df_B.iterrows():
          gene, pathway = row['Gene'], row['Description']
          if gene in gene_to_index and pathway in pathway_to_index:
             gene_idx = gene_to_index[gene]
             pathway_idx = pathway_to_index[pathway]
             adj_matrix[gene_idx, pathway_idx] = 1
             adj_matrix[pathway_idx, gene_idx] = 1  
    file_path=project_config.NETWORK_DIR / 'ReactomePathwaysRelation.txt'
    df_pathway_relationship=pd.read_csv(file_path, sep='\t', header=None,names=['Pathway1', 'Pathway2'])
    for _, row in df_pathway_relationship.iterrows():
        pathway1, pathway2 = row['Pathway1'], row['Pathway2']
        if pathway1 in pathway_to_index and pathway2 in pathway_to_index:
            
            pathway1_idx = pathway_to_index[pathway1]
            pathway2_idx = pathway_to_index[pathway2]
            adj_matrix[pathway1_idx, pathway2_idx] = 1
            adj_matrix[pathway2_idx, pathway1_idx] = 1 
    file_path_g=project_config.NETWORK_DIR / 'genes_edgelist.txt'
    df_gene_relationship=pd.read_csv(file_path_g, sep=' ', header=None,names=['Gene1', 'Gene2'])
    for _, row in df_gene_relationship.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']

        if gene1 in gene_to_index and gene2 in gene_to_index and gene1!=gene2:
            
            gene1_idx = gene_to_index[gene1]
            gene2_idx = gene_to_index[gene2]
            adj_matrix[gene1_idx, gene2_idx] = 1
            adj_matrix[gene2_idx, gene1_idx] = 1  

# Convert to edge_index format.
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

# Optional shape check.
    #print(f"Edge Index Shape: {edge_index.shape}")

    return edge_index






def getedge_index_all_pathway(device,genes_A):
    gmt_file_path = project_config.NETWORK_DIR / 'ReactomePathways.gmt'
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


# Adjacency matrix dimension.
    matrix_size = len(genes_A) + len(involved_pathways)
    adj_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    #print(len(involved_pathways))

# Populate the adjacency matrix.
    for _, row in df_B.iterrows():
          gene, pathway = row['Gene'], row['Description']
          if gene in gene_to_index and pathway in pathway_to_index:
             gene_idx = gene_to_index[gene]
             pathway_idx = pathway_to_index[pathway]
             adj_matrix[gene_idx, pathway_idx] = 1
             adj_matrix[pathway_idx, gene_idx] = 1  
    file_path=project_config.NETWORK_DIR / 'ReactomePathwaysRelation.txt'
    df_pathway_relationship=pd.read_csv(file_path, sep='\t', header=None,names=['Pathway1', 'Pathway2'])
    for _, row in df_pathway_relationship.iterrows():
        pathway1, pathway2 = row['Pathway1'], row['Pathway2']
        if pathway1 in pathway_to_index and pathway2 in pathway_to_index:
            
            pathway1_idx = pathway_to_index[pathway1]
            pathway2_idx = pathway_to_index[pathway2]
            adj_matrix[pathway1_idx, pathway2_idx] = 1
            adj_matrix[pathway2_idx, pathway1_idx] = 1 
    file_path_g=project_config.NETWORK_DIR / 'genes_edgelist.txt'
    df_gene_relationship=pd.read_csv(file_path_g, sep=' ', header=None,names=['Gene1', 'Gene2'])
    for _, row in df_gene_relationship.iterrows():
        gene1, gene2 = row['Gene1'], row['Gene2']

        if gene1 in gene_to_index and gene2 in gene_to_index and gene1!=gene2:
            
            gene1_idx = gene_to_index[gene1]
            gene2_idx = gene_to_index[gene2]
            adj_matrix[gene1_idx, gene2_idx] = 1
            adj_matrix[gene2_idx, gene1_idx] = 1  

# Convert to edge_index format.
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

# Optional shape check.
    #print(f"Edge Index Shape: {edge_index.shape}")

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
