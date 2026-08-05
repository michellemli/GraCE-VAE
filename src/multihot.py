import pandas as pd
import torch
import pickle
import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
from edgeindex import getedge_index_GP,getedge_index_all_pathway
device='cuda:0'
adata = sc.read_h5ad('./cpa_binaries/datasets/Norman2019_raw.h5ad')
genes_A=list(adata.var.gene_symbols)
edgeindex=getedge_index_all_pathway(device,genes_A)
#edgeindex=getedge_index_all(device,genes_A)
datadir='./result/alldata'
device='cuda:0'

with open(f'{datadir}/train_data.pkl', 'rb') as f:
		dataloader2=pickle.load(f)

def onehot(device,X,edge_index):


    n, num_genes = X.shape  # n个细胞，num_genes为5000
    num_pathways = 2694  # pathway 数量
    num_total_features = num_genes + num_pathways  # 总特征数量（基因 + pathway）
    
    # 初始化输出矩阵，形状为 (n, 7000)，前5000维为原始基因数据，后2000维为pathway数据
    result = torch.zeros((n, num_total_features), device=X.device)
    
    # 将原始的基因表达数据保留在结果的前5000个维度
    result[:, :num_genes] = X
    
    # 初始化 pathway 特征的矩阵，形状为 (n, num_pathways)，所有值初始为 0
    pathway_features = torch.zeros((n, num_pathways), device=X.device)
    
    # 遍历所有的边
    for i in range(edge_index.shape[1]):
        if edge_index[0,i]<5000 and edge_index[1, i]>=5000: #choose the edges connecting the genes and the 
            gene_idx = edge_index[0, i]  # 基因索引 0-4999
            pathway_idx = edge_index[1, i] - 5000  # pathway 索引，5000-6999 转换为 0-1999
        
            # 如果该基因在某个细胞中的表达大于 0，设置对应的 pathway feature 为 1
            pathway_features[:, pathway_idx] = torch.max(pathway_features[:, pathway_idx], (X[:, gene_idx] > 0).float())
    
    # 将 pathway 的数据放到结果的后2000维
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