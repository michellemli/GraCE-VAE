import os
import argparse
from argparse import Namespace
import time
import json
import pickle

import numpy as np
import random
import torch
import scanpy as sc
from train import train, train_CVAE, train_MVAE,train_GNN,train_onehot,test_model
from utils import get_data


def infer_embedding_dim(embedding_h5ad, embedding_obsm_key):
	adata = sc.read_h5ad(embedding_h5ad)
	if embedding_obsm_key not in adata.obsm:
		raise KeyError(
			f"Embedding key '{embedding_obsm_key}' not found in {embedding_h5ad}. "
			f"Available keys: {list(adata.obsm.keys())}"
		)
	return int(adata.obsm[embedding_obsm_key].shape[1])



def main(args):
	print(f'using device: {args.device}')

	opts = Namespace(
		batch_size = 32,
		mode = 'train',
		lr = 1e-3,
		epochs = args.epoch,
		grad_clip = args.gradclip,
		randomedge = args.randomedge,
		halfmixededge = args.halfmixededge,
		mxAlpha = args.mxAlpha,
		mxBeta = args.mxBeta,
		mxTemp = args.mxTemp,
		lmbda = args.lmbda,
		lambda2=args.lambda2,
		lambda_interv=args.lambda_interv,
		MMD_sigma = 1000,
		kernel_num = 10,
		matched_IO = False,
		latdim = 105,
		seed = args.randomseed,
		pdim=2694,                           #determin the number of pathway considered, can be 2694(all pathways) or 2082(involved pathway) 
		use_external_embedding=bool(args.embedding_h5ad),
		external_embedding_dim=0,
	)

	torch.manual_seed(opts.seed)
	np.random.seed(opts.seed)
	random.seed(opts.seed)
	
	dataloader, dataloader1, dataloader2, dim, cdim, ptb_targets, embedding_dim = get_data(
		batch_size=opts.batch_size,
		mode=opts.mode,
		embedding_h5ad=args.embedding_h5ad,
		embedding_obsm_key=args.embedding_obsm_key,
		embedding_align_mode=args.embedding_align_mode,
	)
	if args.mode==3:
		opts.pdim=0
	if args.mode==4:
		opts.pdim=2082
	if args.mode==6:
		opts.pdim=2082
	if args.embedding_h5ad:
		opts.external_embedding_dim = embedding_dim if embedding_dim else infer_embedding_dim(args.embedding_h5ad, args.embedding_obsm_key)

	opts.dim = dim
	if opts.latdim is None:
		opts.latdim = cdim
	if args.mode==1:
		opts.latdim=7
	opts.cdim = cdim

	if os.path.exists(args.datadir):
		print(f"Found existing dataset for seed {args.randomseed}. Loading data...")

		with open(f'{args.datadir}/ptb_targets.pkl', 'rb') as f:
			ptb_targets = pickle.load(f)

		with open(f'{args.datadir}/test_data_single_node.pkl', 'rb') as f:
			dataloader2 = pickle.load(f)

		with open(f'{args.datadir}/train_data.pkl', 'rb') as f:
			dataloader = pickle.load(f)

		with open(f'{args.datadir}/val_data_single_node.pkl', 'rb') as f:
			dataloader1 = pickle.load(f)	
	else:
		dataloader, dataloader1, dataloader2, dim, cdim, ptb_targets, embedding_dim = get_data(
			batch_size=opts.batch_size,
			mode=opts.mode,
			embedding_h5ad=args.embedding_h5ad,
			embedding_obsm_key=args.embedding_obsm_key,
			embedding_align_mode=args.embedding_align_mode,
		)
		if args.mode==3:
			opts.pdim=0
		if args.mode==4:
			opts.pdim=2082
		if args.mode==6:
			opts.pdim=2082
		if args.embedding_h5ad:
			opts.external_embedding_dim = embedding_dim if embedding_dim else infer_embedding_dim(args.embedding_h5ad, args.embedding_obsm_key)
		
		print(f"No existing dataset for seed {args.randomseed}. Creating new dataset...")	
		os.makedirs(args.datadir, exist_ok=True)
		with open(f'{args.datadir}/config.json', 'w') as f:
			json.dump(opts.__dict__, f, indent=4)

		with open(f'{args.datadir}/ptb_targets.pkl', 'wb') as f:
			pickle.dump(ptb_targets, f, protocol=pickle.HIGHEST_PROTOCOL)

		with open(f'{args.datadir}/test_data_single_node.pkl', 'wb') as f:
			pickle.dump(dataloader2, f, protocol=pickle.HIGHEST_PROTOCOL)

		with open(f'{args.datadir}/val_data_single_node.pkl', 'wb') as f:
			pickle.dump(dataloader1, f, protocol=pickle.HIGHEST_PROTOCOL)

		with open(f'{args.datadir}/train_data.pkl', 'wb') as f:
			pickle.dump(dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)
    

	if args.model == 'cmvae':
        
		#train(dataloader, dataloader1,ptb_targets,opts, args.device, args.savedir,args.datadir,args.mode, log=args.wandblog)
		model = torch.load(f'{args.savedir}/last_model.pt')
		test_model(model,args.model,args.datadir,args.savedir,ptb_targets,args.device,opts.batch_size,args.mode,args.randomseed, embedding_h5ad=args.embedding_h5ad, embedding_obsm_key=args.embedding_obsm_key)
        
	elif args.model == 'cmvaegnn':
		#print("the latent dimension is{}".format(cdim))
		train_GNN(
			dataloader,
			dataloader1,
			ptb_targets,
			opts,
			args.device,
			args.savedir,
			args.datadir,

			args.mode,
			log=args.wandblog,
			graphencoder=args.graphencoder,
			randomedge=args.randomedge,
			halfmixededge=args.halfmixededge,
			interv_encoder_type=args.interv_encoder_type,
			distill_weight=args.distill_weight,
			distill_output_mode=args.distill_output_mode,
		)
		model = torch.load(f'{args.savedir}/last_model.pt')
		test_model(model,args.model,args.datadir,args.savedir,ptb_targets,args.device,opts.batch_size,args.mode,args.randomseed,args.randomedge,args.halfmixededge, embedding_h5ad=args.embedding_h5ad, embedding_obsm_key=args.embedding_obsm_key)
	    


	elif args.model == 'cmvaeonehot':
		
		train_onehot(dataloader, dataloader1,ptb_targets,opts, args.device, args.savedir, args.datadir,args.mode,log=args.wandblog)
		model = torch.load(f'{args.savedir}/last_model.pt')
		test_model(model,args.model,args.datadir,args.savedir,ptb_targets,args.device,opts.batch_size,args.mode,args.randomseed, embedding_h5ad=args.embedding_h5ad, embedding_obsm_key=args.embedding_obsm_key)
	elif args.model == 'cgvae':
		train_GNN(dataloader, dataloader1,ptb_targets,opts, args.device, args.savedir, args.datadir,args.mode,log=args.wandblog,remove=True)
		model = torch.load(f'{args.savedir}/last_model_{args.graphencoder}.pt')
		test_model(model,args.model,args.datadir,args.savedir,ptb_targets,args.device,opts.batch_size,args.mode,args.randomseed, embedding_h5ad=args.embedding_h5ad, embedding_obsm_key=args.embedding_obsm_key)

	#elif args.model == 'mvae':
		#train_MVAE(dataloader, opts, args.device, args.savedir, log=True) 
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parse args')
	parser.add_argument('-s', '--savedir', type=str, default='./result/', help='directory to save the results')
	parser.add_argument('--device', type=str, default=None, help='device to run the training')
	parser.add_argument('--model', type=str, default="cmvaegnn", help='model to run the training')
	parser.add_argument('--mode', type=int, default=5, help='our model number')
	parser.add_argument('--randomseed', type=int, default=1, help='random seed during the training')
	parser.add_argument('--graphencoder', type=bool, default=False, help='use graph causal encoder or not')
	parser.add_argument('--randomedge', type=bool, default=False, help='shuffle edge order for graph experiment')
	parser.add_argument('--halfmixededge', type=bool, default=False, help='keep 50% true edges and replace 50% with random edges')
        # hyper-parameter finetune  mxAlpha = 10,
	parser.add_argument('--datadir', type=str, default='./alldata/', help='directory to save the data')
	parser.add_argument('--mxAlpha', type=int, default=8, help='mxAlpha')
	parser.add_argument('--mxBeta', type=int, default=2, help='mxBeta')
	parser.add_argument('--mxTemp', type=int, default=4, help='mxTemp')
	parser.add_argument('--lambda2', type=float, default=1e-4, help='lambda2')
	parser.add_argument('--lambda_interv', type=float, default=1e-2, help='regularization for intervention-to-DAG mapping')
	parser.add_argument('--lmbda', type=float, default=1e-4, help='lambda')
	parser.add_argument('--epoch', type=int, default=100, help='trainingepoch')
	parser.add_argument('--gradclip', type=bool, default=False, help='gradclip')
	parser.add_argument('--wandblog', type=bool, default=False, help='use wandb log or not')
	parser.add_argument('--embedding_h5ad', type=str, default=None, help='optional external embedding h5ad aligned to Norman cells')
	parser.add_argument('--embedding_obsm_key', type=str, default='embeddings', help='obsm key to read from embedding_h5ad')
	parser.add_argument('--embedding_align_mode', type=str, default='auto', help='embedding alignment mode: auto | row_order')

	parser.add_argument(
		'--interv_encoder_type',
		type=str,
		default=None,
		help='intervention encoder: v1_trivalue | v2_dropedge | v2_dropedge_distill | v3_target_lookup'
	)
	parser.add_argument(
		'--distill_weight',
		type=float,
		default=0.0,
		help='weight for distillation loss when using v2_dropedge_distill'
	)
	parser.add_argument(
		'--distill_output_mode',
		type=str,
		default='teacher',
		help='for v2_dropedge_distill: teacher | student | mix'
	)
	args = parser.parse_args()
	args.datadir=args.datadir+f'seed{args.randomseed}'
	args.savedir = f"{args.savedir}model{args.mode}_seed{args.randomseed}"
	if args.embedding_h5ad:
		emb_tag = f"_extemb_{args.embedding_obsm_key}"
		args.datadir = f"{args.datadir}{emb_tag}"
		args.savedir = f"{args.savedir}{emb_tag}"
	if args.randomedge:
		args.savedir = f"{args.savedir}_randomedge"
	if args.halfmixededge:
		args.savedir = f"{args.savedir}_halfmixededge"
	if args.graphencoder:
		interv = getattr(args, "interv_encoder_type", "none")

		dmode = getattr(args, "distill_output_mode", "na")
		dwt = getattr(args, "distill_weight", 0.0)

# 只在 v2_dropedge_distill 时记录 distill 细节，避免目录过长
		distill_tag = ""
		if "distill" in str(interv):
			distill_tag = f"_dmode{dmode}_dwt{dwt:g}"
		args.savedir = (
    f"{args.savedir}"
    f"m{args.mode}"
    f"_seed{args.randomseed}"
    f"_enc{interv}"
    f"{distill_tag}"
		)

	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)

	main(args)




'''
	with open(f'{args.datadir}/config.json', 'w') as f:
		json.dump(opts.__dict__, f, indent=4)

	with open(f'{args.datadir}/ptb_targets.pkl', 'wb') as f:
		pickle.dump(ptb_targets, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open(f'{args.datadir}/test_data_single_node.pkl', 'wb') as f:
		pickle.dump(dataloader2, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open(f'{args.datadir}/train_data.pkl', 'wb') as f:
		pickle.dump(dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)
'''
