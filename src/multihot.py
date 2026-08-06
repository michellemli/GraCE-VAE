import torch
import pickle
import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
from edgeindex import getedge_index_all_pathway

import project_config


device='cuda:0'
adata = sc.read_h5ad(project_config.SCDATA)
genes_A=list(adata.var.gene_symbols)
edgeindex=getedge_index_all_pathway(device,genes_A)
#edgeindex=getedge_index_all(device,genes_A)
datadir='./result/alldata'
device='cuda:0'

with open(f'{datadir}/train_data.pkl', 'rb') as f:
		dataloader2=pickle.load(f)

def onehot(device,X,edge_index):


    n, num_genes = X.shape  # n cells and 5,000 genes
    num_pathways = 2694
    num_total_features = num_genes + num_pathways
    
    # Store gene expression first and pathway indicators second.
    result = torch.zeros((n, num_total_features), device=X.device)
    
    # Preserve the original gene expression values.
    result[:, :num_genes] = X
    
    # Initialize binary pathway features.
    pathway_features = torch.zeros((n, num_pathways), device=X.device)
    
    # Visit every gene-pathway edge.
    for i in range(edge_index.shape[1]):
        if edge_index[0,i]<5000 and edge_index[1, i]>=5000: #choose the edges connecting the genes and the 
            gene_idx = edge_index[0, i]  # Gene index 0-4999
            pathway_idx = edge_index[1, i] - 5000
        
            # Activate a pathway if any member gene is expressed.
            pathway_features[:, pathway_idx] = torch.max(pathway_features[:, pathway_idx], (X[:, gene_idx] > 0).float())
    
    # Store pathway values after the gene dimensions.
    result[:, num_genes:] = pathway_features
    
    return pathway_features
all_X4=[]
for (i, X) in enumerate(dataloader2):
            x = X[0].cuda()#observation samples

            dx=onehot(device,x,edgeindex).unsqueeze(0)
            all_X4.append(dx)

            print(i)
all_X4_tensor=torch.cat(all_X4, dim=0)
save_path='./result/alldata/multi-hot-train-allpathways'
torch.save(all_X4_tensor, save_path)
print(f"Preprocessed data saved to {save_path}")
