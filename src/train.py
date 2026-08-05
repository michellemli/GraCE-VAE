import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from copy import deepcopy
import numpy as np
import os
from model import CMVAEGNN, CMVAE, CMVAEonehot, CGVAE, GRACE
from edgeindex import getedge_index_GGGP,getedge_index_all,getedge_index_GG,getedge_index_all_pathway,build_gene_union_graph,mixed_half_correct_half_random_index_all_pathway
import scanpy as sc
from sklearn.metrics import r2_score
from inference import evaluate_generated_samples, evaluate_single_leftout, evaluate_double
from utils import MMD_loss, laplacian_smoothness

import project_config


def _get_batch_embedding(batch):
    if len(batch) > 3:
        return batch[3]
    return None


# fit CMVAE to data
def train(
dataloader,val_dataloader,ptb_targets,
    opts,
    device,
    savedir,datadir,
    mode,
    log,
    ):
    
    if log:
        wandb.init(project='cmvae', name=savedir.split('/')[-1])  

    cmvae = CMVAE(
        dim = opts.dim,
        z_dim = opts.latdim,
        c_dim = opts.cdim,
        device = device
    )
    cmvae.double()
    cmvae.to(device)

    optimizer = torch.optim.Adam(params=cmvae.parameters(), lr=opts.lr)

    cmvae.train()
    print("Training for {} epochs...".format(str(opts.epochs)))
    
    ## Loss parameters
    if opts.epochs == 1:
        beta_schedule = torch.tensor([0.0])
        alpha_schedule = torch.tensor([0.0])
        temp_schedule = torch.tensor([1.0])
    else:
        beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
        beta_schedule[:10] = 0
        beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10) 
        alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
        alpha_schedule[:] = opts.mxAlpha
        alpha_schedule[:5] = 0
        alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
        alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

        ## Softmax temperature 
        temp_schedule = torch.ones(opts.epochs)
        temp_schedule[5:] = torch.linspace(1, opts.mxTemp, opts.epochs-5)
   
    min_train_loss = np.inf
    best_model = deepcopy(cmvae)
    print('the architecture is:',cmvae)
    for n in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0
        L1Av = 0
        
        for (i, X) in enumerate(dataloader):
            x = X[0]#observation samples
            y = X[1]#interventional samples
            c = X[2]#intervention
            
            if cmvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)

            optimizer.zero_grad()
            y_hat, x_recon, z_mu, z_var, G = cmvae(x, c, c, num_interv=1, temp=temp_schedule[n])# add edgeindex for GNN
            mmd_loss, recon_loss, kl_loss, L1 = loss_function(y_hat, y, x_recon, x, z_mu, z_var, G, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss + opts.lmbda*L1
            loss.backward()
            if opts.grad_clip:
                for param in cmvae.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            mmdAv += mmd_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            klAv += kl_loss.detach().cpu().numpy()
            L1Av += L1.detach().cpu().numpy()

            if log:
                wandb.log({'loss':loss})
                wandb.log({'mmd_loss':mmd_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})

        print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg mmd_loss': mmdAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})

        if (mmdAv + reconAv + klAv + L1Av)/ct < min_train_loss:
            min_train_loss = (mmdAv + reconAv + klAv + L1Av)/ct 
            best_model = deepcopy(cmvae)
            #torch.save(best_model, os.path.join(savedir, 'best_model.pt'))

    last_model = deepcopy(cmvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))
    print("model saved!")
    loss_fn = MMD_loss(fix_sigma=1000, kernel_num=10).cuda()
    adata=sc.read_h5ad(project_config.SCDATA)
    cmvae.eval()
    
    print("start hyper parameter finetune on the validation  data-set")
    rmse, signerr, gt_y, pred_y, c_y, gt_x = evaluate_generated_samples(cmvae,val_dataloader,device,temp=1,modelnumber=mode,numint=1,mode="cmvae") 
    metric={}
    C_y = [','.join([str(l) for l in np.where(c_y[i] != 0)[0]]) for i in range(c_y.shape[0])]
    
    # Initialize lists for key metrics
    rmse_int, rsquare_int, mmd_int = [], [], []
    rmse_deg, rsquare_deg, mmd_deg = [], [], []

    for i in tqdm(set(C_y)):
      i=int(i)

      gt = gt_y[np.where(c_y[:,i]!=0)[0],:]
      pred = pred_y[np.where(c_y[:,i]!=0)[0],:]
      rmse_int.append(np.sqrt(np.mean(((pred[:] - gt[:])**2)) / np.mean(((gt[:])**2))))
      rsquare_int.append(max(r2_score(np.mean(pred, axis=0),np.mean(gt, axis=0)),0))
      y_hat=torch.tensor(gt[:])
      y=torch.tensor(pred[:])
      mmd_int.append(loss_fn(y_hat, y).item())
      g1 = ptb_targets[i]
      for n,name in enumerate(adata.uns['rank_genes_groups']['names'].dtype.names):
        if g1 in name:
            dg = [x[n] for x in adata.uns['rank_genes_groups']['names']]
            break
    
      idx = [list(adata.var.index).index(deg) for deg in dg]
    

      rmse_deg.append(np.sqrt(np.mean(((pred[:,idx] - gt[:,idx])**2)) / np.mean(((gt[:,idx])**2))))
      rsquare_deg.append(max(r2_score(np.mean(pred[:,idx], axis=0),np.mean(gt[:,idx], axis=0)),0))
      degy_hat=torch.tensor(pred[:,idx]).cuda()
      degy=torch.tensor(gt[:,idx]).cuda()
      mmd_deg.append(loss_fn(degy_hat, degy).item())

    # Summarize single evaluation key metrics
    metric= { 'mxAlpha': opts.mxAlpha  , 'mxBeta':opts.mxBeta, 'mxTemp': opts.mxTemp,'lambda': opts.lmbda, 'grad_clip': opts.grad_clip,
        'R2': np.mean(rsquare_deg),        'RMSE':  np.mean(rmse_deg),        'MMD': np.mean(mmd_deg)    }
    filename = os.path.join(datadir, 'finetune_model_{}_results.txt'.format(mode))
    
    # Prepare a line to write to the file
    result_line = [
        metric['mxAlpha'],
        metric['mxBeta'],
        metric['mxTemp'],
        metric['lambda'],
        metric['grad_clip'],
        metric['R2'],
        metric['RMSE'],
        metric['MMD'],
        mode   # Include mode if needed
    ]
    
    # Write header if the file does not exist yet
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as file:
            header = [
                'mxAlpha', 'mxBeta', 'mxTemp', 'lambda', 'grad_clip',
                'R2', 'RMSE', 'MMD', 'Mode'
            ]
            file.write('\t'.join(header) + '\n')

    # Append results for each run
    with open(filename, "a", encoding="utf-8") as file:
        file.write('\t'.join(map(str, result_line)) + '\n')
    
    print(f"validation set results saved to {filename}")
    
    last_model = deepcopy(cmvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))
    print("model saved!")
# loss function definition
def loss_function(y_hat, y, x_recon, x, mu, var, G, MMD_sigma, kernel_num, matched_IO=False):

    if not matched_IO:
        matching_function_interv = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num) # MMD Distance since we don't have paired data
    else:
        matching_function_interv = nn.MSELoss() # MSE if there is matched interv/observ samples
    matching_function_recon = nn.MSELoss() # reconstruction

    if y_hat is None:
        MMD = 0
    else:
        MMD = matching_function_interv(y_hat, y)
    MSE = matching_function_recon(x_recon, x)
    logvar = torch.log(var)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)/x.shape[0]
    if G is None:
        L1 = 0
    else:
        L1 = torch.norm(torch.triu(G,diagonal=1),1)  # L1 norm for sparse G
    return MMD, MSE, KLD, L1


#fit GNN to data
def train_GNN(
    dataloader, val_dataloader,ptb_targets,
    opts,
    device,
    savedir,datadir,
    mode,
    log,
    remove=False,
    graphencoder=False,
    halfmixededge=False,
    interv_encoder_type=None):
    
    if log:
        wandb.init(project='cmvae', name=savedir.split('/')[-1])  

    adata = sc.read_h5ad(project_config.SCDATA)
    genes_A = list(adata.var.gene_symbols)

    if remove:
        cmvae = CGVAE(
            dim = opts.dim,
            z_dim = opts.latdim,
            c_dim = opts.cdim,
            device = device,
            p_dim=opts.pdim,
            mode=mode
        ) 
    elif graphencoder:
        adata = sc.read_h5ad(project_config.SCDATA)
        genes_X = list(adata.var.gene_symbols)  
        genes_all, interv_edge_index, ptb_to_node = build_gene_union_graph(
            device=device,
            genes_X=genes_X,
            ptb_targets=ptb_targets
        )
        interv_num_nodes = len(genes_all)
        cmvae = GRACE(
            dim=opts.dim,
            z_dim=opts.latdim,
            c_dim=opts.cdim,
            p_dim=opts.pdim,
            mode=mode,
            device=device,
            interv_num_nodes=interv_num_nodes,
            interv_edge_index=interv_edge_index,
            ptb_to_node=ptb_to_node,
            node_emb_dim=32,
            gnn_hidden=64,
            readout="target",
            interv_encoder_type=interv_encoder_type,
        )
    else:      
        cmvae = CMVAEGNN(
            dim = opts.dim,
            z_dim = opts.latdim,
            c_dim = opts.cdim,
            device = device,
            p_dim=opts.pdim,
            mode=mode,
            use_external_embedding=getattr(opts, "use_external_embedding", False),
            external_embedding_dim=getattr(opts, "external_embedding_dim", 0),
        )#add p_dim for CMVAEGNN
    cmvae.double()
    cmvae.to(device)

    optimizer = torch.optim.Adam(params=cmvae.parameters(), lr=opts.lr)

    cmvae.train()
    print("Training for {} epochs...".format(str(opts.epochs)))
    
    ## Loss parameters
    if opts.epochs==1:
        beta_schedule = torch.tensor([0.0])
        alpha_schedule = torch.tensor([0.0])
        temp_schedule = torch.tensor([1.0])
    else:
        beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
    
        beta_schedule[:10] = 0
        beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10) 
        alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
        alpha_schedule[:] = opts.mxAlpha
        alpha_schedule[:5] = 0
        alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
        alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha
    

    ## Softmax temperature 
        temp_schedule = torch.ones(opts.epochs)
        temp_schedule[5:] = torch.linspace(1, opts.mxTemp, opts.epochs-5)
   
    min_train_loss = np.inf
    #best_model = deepcopy(cmvae)

    print("\nModel Parameters for GRACE:")
    for name, param in cmvae.named_parameters():
          print(f"{name}: {param.shape}")
    total_params = sum(p.numel() for p in cmvae.parameters())
    print(f"Total number of parameters: {total_params}")
    if graphencoder or remove:
        edgeindex=getedge_index_all_pathway(device,genes_A)
        if mode==3:
            edgeindex=getedge_index_GG(device,genes_A)
        if mode==4:
            edgeindex=getedge_index_GGGP(device,genes_A)
        if mode==6:
            edgeindex=getedge_index_all(device,genes_A)
    else:
        if halfmixededge:
            edgeindex=mixed_half_correct_half_random_index_all_pathway(device,genes_A,seed=opts.seed)
        else:
            edgeindex=getedge_index_all_pathway(device,genes_A)
            if mode==3:
                edgeindex=getedge_index_GG(device,genes_A)
            if mode==4:
                edgeindex=getedge_index_GGGP(device,genes_A)
            if mode==6:
                edgeindex=getedge_index_all(device,genes_A)
    print("the size of egde in GNN is {}".format(edgeindex.shape[1]/2))
    onehot=0#torch.load(f'./alldata/seed{opts.seed}/multi-hot-train-allpathways')
    ggindex=getedge_index_GG(device,genes_A)
    for n in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0
        L1Av = 0
        for (i, X) in enumerate(dataloader):
            x = X[0]#observation samples
            y = X[1]#interventional samples
            c = X[2]#intervention
            dx = _get_batch_embedding(X)
            if dx is None:
                dx = 0
            if cmvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                if isinstance(dx, torch.Tensor):
                    dx = dx.to(device)
            optimizer.zero_grad()
            if graphencoder:
                y_hat, x_recon, z_mu, z_var, G,bc,csz,bc2,csz2 = cmvae(x,dx,c, c, edgeindex,mode,num_interv=1, temp=temp_schedule[n])
            else:
                y_hat, x_recon, z_mu, z_var, G = cmvae(x,dx,c, c, edgeindex,mode,num_interv=1, temp=temp_schedule[n])
            mmd_loss, recon_loss, kl_loss, L1 = loss_function(y_hat, y, x_recon, x, z_mu, z_var, G, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            if graphencoder or remove:
                if hasattr(cmvae, "pathway_graph_regularization"):
                    loss_smooth = cmvae.pathway_graph_regularization()
                else:
                    W = cmvae.d2.weight
                    loss_smooth = laplacian_smoothness(W, ggindex, edge_weight=None)
                loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss + opts.lmbda*L1 + opts.lambda2*loss_smooth
            else:
                loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss + opts.lmbda*L1

            loss.backward()
            if opts.grad_clip:
                for param in cmvae.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            mmdAv += mmd_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            klAv += kl_loss.detach().cpu().numpy()
            L1Av += L1.detach().cpu().numpy()

            if log:
                wandb.log({'loss':loss})
                wandb.log({'mmd_loss':mmd_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})

        

        print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg mmd_loss': mmdAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})
        if log and graphencoder:
            with torch.no_grad():
                bc_ = bc.detach()
                eta_ = csz.detach()

                bc_entropy = (-(bc_ * (bc_ + 1e-8).log()).sum(dim=1)).mean().item()
                bc_top1 = bc_.max(dim=1).values.mean().item()
                bc_top5 = torch.topk(bc_, k=min(5, bc_.size(1)), dim=1).values.sum(dim=1).mean().item()

                eta_mean = eta_.mean().item()
                eta_abs = eta_.abs().mean().item()
                eta_std = eta_.std().item()
                eta_pos_frac = (eta_ > 0).float().mean().item()

            wandb.log({
        "bc_entropy": bc_entropy,
        "bc_top1": bc_top1,
        "bc_top5": bc_top5,
        "eta_mean": eta_mean,
        "eta_abs": eta_abs,
        "eta_std": eta_std,
        "eta_pos_frac": eta_pos_frac,
            })

        #if (mmdAv + reconAv + klAv + L1Av)/ct < min_train_loss:
            #min_train_loss = (mmdAv + reconAv + klAv + L1Av)/ct 
            #best_model = deepcopy(cmvae)
            #torch.save(best_model, os.path.join(savedir, 'best_model.pt'))
    # Final validation loss calculation
    loss_fn = MMD_loss(fix_sigma=1000, kernel_num=10).cuda()
    adata=sc.read_h5ad(project_config.SCDATA)
    cmvae.eval()
    
    print("start hyper parameter finetune on the validation  data-set")
    rmse, signerr, gt_y, pred_y, c_y, gt_x = evaluate_generated_samples(cmvae,val_dataloader,device,temp=1,modelnumber=mode,numint=1,mode="cmvaegnn", halfmixededge=halfmixededge, seed=opts.seed)
    metric={}
    C_y = [','.join([str(l) for l in np.where(c_y[i] != 0)[0]]) for i in range(c_y.shape[0])]
    
    # Initialize lists for key metrics
    rmse_int, rsquare_int, mmd_int = [], [], []
    rmse_deg, rsquare_deg, mmd_deg = [], [], []

    for i in tqdm(set(C_y)):
      i=int(i)

      gt = gt_y[np.where(c_y[:,i]!=0)[0],:]
      pred = pred_y[np.where(c_y[:,i]!=0)[0],:]
      rmse_int.append(np.sqrt(np.mean(((pred[:] - gt[:])**2)) / np.mean(((gt[:])**2))))
      rsquare_int.append(max(r2_score(np.mean(pred, axis=0),np.mean(gt, axis=0)),0))
      y_hat=torch.tensor(gt[:])
      y=torch.tensor(pred[:])
      mmd_int.append(loss_fn(y_hat, y).item())
      g1 = ptb_targets[i]
      for n,name in enumerate(adata.uns['rank_genes_groups']['names'].dtype.names):
        if g1 in name:
            dg = [x[n] for x in adata.uns['rank_genes_groups']['names']]
            break
    
      idx = [list(adata.var.index).index(deg) for deg in dg]
    

      rmse_deg.append(np.sqrt(np.mean(((pred[:,idx] - gt[:,idx])**2)) / np.mean(((gt[:,idx])**2))))
      rsquare_deg.append(max(r2_score(np.mean(pred[:,idx], axis=0),np.mean(gt[:,idx], axis=0)),0))
      degy_hat=torch.tensor(pred[:,idx]).cuda()
      degy=torch.tensor(gt[:,idx]).cuda()
      mmd_deg.append(loss_fn(degy_hat, degy).item())

    # Summarize single evaluation key metrics
    metric= { 'mxAlpha': opts.mxAlpha  , 'mxBeta':opts.mxBeta, 'mxTemp': opts.mxTemp,'lambda': opts.lmbda, 'grad_clip': opts.grad_clip,
        'R2': np.mean(rsquare_deg),        'RMSE':  np.mean(rmse_deg),        'MMD': np.mean(mmd_deg)    }
    filename = os.path.join(savedir, 'finetune_model_results.txt'.format(mode))
    
    # Prepare a line to write to the file
    result_line = [
        metric['mxAlpha'],
        metric['mxBeta'],
        metric['mxTemp'],
        metric['lambda'],
        metric['grad_clip'],
        metric['R2'],
        metric['RMSE'],
        metric['MMD'],
        mode   # Include mode if needed
    ]
    
    # Write header if the file does not exist yet
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as file:
            header = [
                'mxAlpha', 'mxBeta', 'mxTemp', 'lambda', 'grad_clip',
                'R2', 'RMSE', 'MMD', 'Mode'
            ]
            file.write('\t'.join(header) + '\n')

    # Append results for each run
    with open(filename, "a", encoding="utf-8") as file:
        file.write('\t'.join(map(str, result_line)) + '\n')
    
    print(f"validation set results saved to {filename}")
    
    last_model = deepcopy(cmvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))
    print("model saved!")
def train_onehot(
    dataloader,val_dataloader,ptb_targets,
    opts,
    device,
    savedir,datadir,
    mode,
    log,
    ):
    
    if log:
        wandb.init(project='cmvae', name=savedir.split('/')[-1])  

    cmvae = CMVAEonehot(
        dim = opts.dim,
        z_dim = opts.latdim,
        c_dim = opts.cdim,
        device = device,
        p_dim=opts.pdim
    )
    cmvae.double()
    cmvae.to(device)

    optimizer = torch.optim.Adam(params=cmvae.parameters(), lr=opts.lr)

    cmvae.train()
    print("Training for {} epochs...".format(str(opts.epochs)))
    
    ## Loss parameters
    if opts.epochs == 1:
        beta_schedule = torch.tensor([0.0])
        alpha_schedule = torch.tensor([0.0])
        temp_schedule = torch.tensor([1.0])
    else:
        beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
        beta_schedule[:10] = 0
        beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10) 
        alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
        alpha_schedule[:] = opts.mxAlpha
        alpha_schedule[:5] = 0
        alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
        alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

        ## Softmax temperature 
        temp_schedule = torch.ones(opts.epochs)
        temp_schedule[5:] = torch.linspace(1, opts.mxTemp, opts.epochs-5)
   
    min_train_loss = np.inf
    best_model = deepcopy(cmvae)
    print(cmvae)
    onehot = None
    legacy_multihot_path = f'./alldata/seed{opts.seed}/multi-hot-train-allpathways'
    if os.path.exists(legacy_multihot_path):
        onehot = torch.load(legacy_multihot_path)
    #edgeindex=getedge_index_all_pathway(device,genes_A)
    #print("the size of egde in GNN is {}".format(edgeindex.shape[1]/2))
    for n in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0
        L1Av = 0

        for (i, X) in enumerate(dataloader):

            x = X[0]#observation samples
            y = X[1]#interventional samples
            c = X[2]#intervention
            dx = _get_batch_embedding(X)
            if dx is None:
                if onehot is None:
                    raise RuntimeError(
                        "No batch-aligned embedding found in the dataloader and no legacy "
                        f"embedding file found at {legacy_multihot_path}."
                    )
                dx = onehot[i, :, :]
            
            
            if cmvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                dx=dx.to(device)
                
            
            optimizer.zero_grad()
            y_hat, x_recon, z_mu, z_var, G = cmvae(x, dx,c, c,mode,num_interv=1, temp=temp_schedule[n])
            mmd_loss, recon_loss, kl_loss, L1 = loss_function(y_hat, y, x_recon, x, z_mu, z_var, G, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss + opts.lmbda*L1
            loss.backward()
            if opts.grad_clip:
                for param in cmvae.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            mmdAv += mmd_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            klAv += kl_loss.detach().cpu().numpy()
            L1Av += L1.detach().cpu().numpy()

            if log:
                wandb.log({'loss':loss})
                wandb.log({'mmd_loss':mmd_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})

        print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg mmd_loss': mmdAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})

        if (mmdAv + reconAv + klAv + L1Av)/ct < min_train_loss:
            min_train_loss = (mmdAv + reconAv + klAv + L1Av)/ct 
            best_model = deepcopy(cmvae)
            #torch.save(best_model, os.path.join(savedir, 'best_model.pt'))
    # Final validation loss calculation
    loss_fn = MMD_loss(fix_sigma=1000, kernel_num=10).cuda()
    adata=sc.read_h5ad(project_config.SCDATA)
    
    cmvae.eval()
    
    print("start hyper parameter finetune on the validation  data-set")
    rmse, signerr, gt_y, pred_y, c_y, gt_x = evaluate_generated_samples(cmvae,val_dataloader,device,temp=1,modelnumber=mode,numint=1,mode="cmvaeonehot") 
    metric={}
    C_y = [','.join([str(l) for l in np.where(c_y[i] != 0)[0]]) for i in range(c_y.shape[0])]
    
    # Initialize lists for key metrics
    rmse_int, rsquare_int, mmd_int = [], [], []
    rmse_deg, rsquare_deg, mmd_deg = [], [], []

    for i in tqdm(set(C_y)):
      i=int(i)

      gt = gt_y[np.where(c_y[:,i]!=0)[0],:]
      pred = pred_y[np.where(c_y[:,i]!=0)[0],:]
      rmse_int.append(np.sqrt(np.mean(((pred[:] - gt[:])**2)) / np.mean(((gt[:])**2))))
      rsquare_int.append(max(r2_score(np.mean(pred, axis=0),np.mean(gt, axis=0)),0))
      y_hat=torch.tensor(gt[:])
      y=torch.tensor(pred[:])
      mmd_int.append(loss_fn(y_hat, y).item())
      g1 = ptb_targets[i]
      for n,name in enumerate(adata.uns['rank_genes_groups']['names'].dtype.names):
        if g1 in name:
            dg = [x[n] for x in adata.uns['rank_genes_groups']['names']]
            break
    
      idx = [list(adata.var.index).index(deg) for deg in dg]
    

      rmse_deg.append(np.sqrt(np.mean(((pred[:,idx] - gt[:,idx])**2)) / np.mean(((gt[:,idx])**2))))
      rsquare_deg.append(max(r2_score(np.mean(pred[:,idx], axis=0),np.mean(gt[:,idx], axis=0)),0))
      degy_hat=torch.tensor(pred[:,idx]).cuda()
      degy=torch.tensor(gt[:,idx]).cuda()
      mmd_deg.append(loss_fn(degy_hat, degy).item())

    # Summarize single evaluation key metrics
    metric= { 'mxAlpha': opts.mxAlpha  , 'mxBeta':opts.mxBeta, 'mxTemp': opts.mxTemp,'lambda': opts.lmbda, 'grad_clip': opts.grad_clip,
        'R2': np.mean(rsquare_deg),        'RMSE':  np.mean(rmse_deg),        'MMD': np.mean(mmd_deg)    }
    filename = os.path.join(datadir, 'finetune_model_{}_results.txt'.format(mode))
    
    # Prepare a line to write to the file
    result_line = [
        metric['mxAlpha'],
        metric['mxBeta'],
        metric['mxTemp'],
        metric['lambda'],
        metric['grad_clip'],
        metric['R2'],
        metric['RMSE'],
        metric['MMD'],
        mode   # Include mode if needed
    ]
    
    # Write header if the file does not exist yet
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as file:
            header = [
                'mxAlpha', 'mxBeta', 'mxTemp', 'lambda', 'grad_clip',
                'R2', 'RMSE', 'MMD', 'Mode'
            ]
            file.write('\t'.join(header) + '\n')

    # Append results for each run
    with open(filename, "a", encoding="utf-8") as file:
        file.write('\t'.join(map(str, result_line)) + '\n')
    
    print(f"validation set results saved to {filename}")
    
    last_model = deepcopy(cmvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))
    print("model saved!")



def test_model(
    model,
    mode,
    datadir,
    savedir,
    ptb_targets,
    device='cuda:0',
    batch_size=32,
    modelnumber=0,
    seed=0,
    halfmixededge=False,
    embedding_h5ad=None,
    embedding_obsm_key='embeddings',
):
    """
    Evaluate the model using both single and double variable evaluations, and summarize key metrics.

    Args:
        model: Trained model to be evaluated.
        datadir (str): Directory containing the data files.
        ptb_targets (list): List of perturbation targets.
        adata (AnnData): AnnData object containing gene data.
        mode (str): Model/inference type.
        batch_size (int): Batch size for evaluation (default: 32).

    Returns:
        dict: A dictionary containing the key evaluation metrics for both single and double evaluations.
    """
    metrics = {}

    # Initialize the MMD loss function
    loss_fn = MMD_loss(fix_sigma=1000, kernel_num=10).cuda()
    adata=sc.read_h5ad(project_config.SCDATA)
    if hasattr(model, "graph_c_encoder") and hasattr(model.graph_c_encoder, "output_mode"):
        model.graph_c_encoder.output_mode = "student"
    model.eval()
    # Step 1: Single Evaluation
    print("Evaluating single...")
    rmse, signerr, gt_y, pred_y, c_y, gt_x = evaluate_single_leftout(model, datadir, model.device, mode,modelnumber, halfmixededge=halfmixededge, seed=seed)
    C_y = [','.join([str(l) for l in np.where(c_y[i] != 0)[0]]) for i in range(c_y.shape[0])]
    
    # Initialize lists for key metrics
    rmse_int, rsquare_int, mmd_int = [], [], []
    rmse_deg, rsquare_deg, mmd_deg = [], [], []

    for i in tqdm(set(C_y)):
      i=int(i)

      gt = gt_y[np.where(c_y[:,i]!=0)[0],:]
      pred = pred_y[np.where(c_y[:,i]!=0)[0],:]
      rmse_int.append(np.sqrt(np.mean(((pred[:] - gt[:])**2)) / np.mean(((gt[:])**2))))
      rsquare_int.append(max(r2_score(np.mean(pred, axis=0),np.mean(gt, axis=0)),0))
      y_hat=torch.tensor(gt[:])
      y=torch.tensor(pred[:])
      mmd_int.append(loss_fn(y_hat, y).item())
      g1 = ptb_targets[i]
      for n,name in enumerate(adata.uns['rank_genes_groups']['names'].dtype.names):
        if g1 in name:
            dg = [x[n] for x in adata.uns['rank_genes_groups']['names']]
            break
    
      idx = [list(adata.var.index).index(deg) for deg in dg]
    

      rmse_deg.append(np.sqrt(np.mean(((pred[:,idx] - gt[:,idx])**2)) / np.mean(((gt[:,idx])**2))))
      rsquare_deg.append(max(r2_score(np.mean(pred[:,idx], axis=0),np.mean(gt[:,idx], axis=0)),0))
      degy_hat=torch.tensor(pred[:,idx]).cuda()
      degy=torch.tensor(gt[:,idx]).cuda()
      mmd_deg.append(loss_fn(degy_hat, degy).item())

    # Summarize single evaluation key metrics
    metrics['Single Evaluation'] = {
        'R2': {'DE Genes': np.mean(rsquare_deg), 'All Genes': np.mean(rsquare_int)},
        'RMSE': {'DE Genes': np.mean(rmse_deg), 'All Genes': np.mean(rmse_int)},
        'MMD': {'DE Genes': np.mean(mmd_deg), 'All Genes': np.mean(mmd_int)}
    }

    # Step 2: Double Evaluation
    print("Evaluating double...")
    rmse, signerr, gt_y, pred_y, c_y, gt_x = evaluate_double(
        model,
        datadir,
        model.device,
        mode,
        modelnumber,
        halfmixededge=halfmixededge,
        seed=seed,
        embedding_h5ad=embedding_h5ad,
        embedding_obsm_key=embedding_obsm_key,
    )
    C_y = [','.join([str(l) for l in np.where(c_y[i] != 0)[0]]) for i in range(c_y.shape[0])]
    
    # Re-initialize lists for double evaluation key metrics
    rmse_int = []
    rsquare_int = []
    mmd_int=[]
    rmse_deg = []
    rsquare_deg = []
    mmd_deg=[]
    loss = MMD_loss(fix_sigma=100, kernel_num=10).cuda()

    for i in tqdm(set(C_y)):
      k = int(i.split(',')[0])
      l = int(i.split(',')[1])

      gt = gt_y[np.where(c_y[:,k]*c_y[:,l]!=0)[0],:]
      pred = pred_y[np.where(c_y[:,k]*c_y[:,l]!=0)[0],:]
      rmse_int.append(np.sqrt(np.mean(((pred[:] - gt[:])**2)) / np.mean(((gt[:])**2))))
      rsquare_int.append(max(r2_score(np.mean(pred, axis=0),np.mean(gt, axis=0)),0))
      y_hat=torch.tensor(gt[:]).cuda()
      y=torch.tensor(pred[:]).cuda()
      #mmd_int.append(loss(y_hat, y).item())
      g1 = ptb_targets[k]
      g2 = ptb_targets[l]
      for n,name in enumerate(adata.uns['rank_genes_groups']['names'].dtype.names):
        if g1 in name and g2 in name:
            dg = [x[n] for x in adata.uns['rank_genes_groups']['names']]
            break
      idx = [list(adata.var.index).index(deg) for deg in dg]
    

      rmse_deg.append(np.sqrt(np.mean(((pred[:,idx] - gt[:,idx])**2)) / np.mean(((gt[:,idx])**2))))
      rsquare_deg.append(max(r2_score(np.mean(pred[:,idx], axis=0),np.mean(gt[:,idx], axis=0)),0))
      degy_hat=torch.tensor(gt[:,idx])
      degy=torch.tensor(pred[:,idx])
      mmd_deg.append(loss(degy_hat, degy).item())
    # Summarize double evaluation key metrics

    mmd_loss = {}
    for i in range(len(C_y) // 32):

    # Move data to the GPU
      y = torch.from_numpy(gt_y[i*32:(i+1)*32]).cuda()
      y_hat = torch.from_numpy(pred_y[i*32:(i+1)*32]).cuda()
      c = C_y[i*32]
    
    # Calculate the losses and metrics
      if c in mmd_loss.keys():
        mmd_loss[c].append(loss_fn(y_hat, y).item())

      else:
        mmd_loss[c] = [loss_fn(y_hat, y).item()] 
       
        
# Summarize statistics
    mmd_loss_summary = {}
  

    for k in tqdm(mmd_loss.keys()):
      mmd_loss_summary[k] = (np.average(mmd_loss[k]), np.std(mmd_loss[k]))
    
    metrics['Double Evaluation'] = {
        'MMD': {'DE Genes': np.mean(mmd_deg),"Mean within Batches": np.mean([i[0] for i in mmd_loss_summary.values()])},
        'R2': {'DE Genes': np.mean(rsquare_deg), 'All Genes': np.mean(rsquare_int)},
        'RMSE': {'DE Genes': np.mean(rmse_deg), 'All Genes': np.mean(rmse_int)}
    }
    filename=os.path.join(savedir, 'model{}_seed{}.txt'.format(modelnumber,seed))
    with open(filename, "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    return metrics
