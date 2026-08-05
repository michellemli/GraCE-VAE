# GraCE-VAE: Causal Representation Learning from Network Data

**Authors:** Jifan Zhang*, Michelle M. Li*, Elena Zheleva

**Paper:** [Causal Representation Learning from Network Data](https://doi.org/10.1145/3770855.3819023)

## Overview

GraCE-VAE is a graph-aware causal discrepancy variational autoencoder for predicting single-cell responses to unseen interventions. This repository contains the Norman CRISPR dataset implementation used in the paper. Pathway nodes provide auxiliary context in the GNN encoder; the causal decoder remains a latent structural causal model.

## Included Models

| CLI name | Paper model |
| --- | --- |
| `cmvaegnn` | GraCE-VAE |
| `cmvae` | CMVAE |
| `cmvaeonehot` | CMVAE-multihot |
| `cgvae` | VGAE |

The SENA baseline uses its separate upstream implementation and is not duplicated here. Experimental model variants that are not reported in the paper have been removed.

## Setup

```bash
git clone https://github.com/michellemli/GraCE-VAE.git
cd GraCE-VAE
conda env create -f environment.yml
conda activate GraCE_env
```

Place the Norman dataset at:

```text
data/Norman2019_raw.h5ad
```

The loader expects `adata.obs['guide_ids']`, `adata.var.gene_symbols`, and the differential-expression results in `adata.uns['rank_genes_groups']`. The Reactome and gene-network files required by the graph encoder are included under `networks/`.

## Main Experiment

```bash
python -W ignore src/run.py \
  --model cmvaegnn \
  --device cuda:0 \
  --mode 15 \
  --randomseed 1 \
  --mxAlpha 8 \
  --mxBeta 2 \
  --mxTemp 4 \
  --lmbda 0.0001
```

`bash run.sh` runs this configuration for seeds 1 through 10.

## Reported Ablations

The retained `--mode` values reproduce the graph-context and GNN architecture variants used in the paper:

| Mode | Encoder graph / architecture |
| --- | --- |
| `3` | Gene-gene graph |
| `4` | Gene-gene and gene-pathway graph |
| `6` | Gene-gene, gene-pathway, and pathway-pathway graph over involved pathways |
| `5` | Full biological graph with one-layer GCN |
| `11` | Full biological graph with one-layer GAT |
| `13` | Full biological graph with three-layer GAT |
| `15` | Full biological graph with one-layer GraphSAGE (main model) |

Use the 50% edge-corruption experiment with:

```bash
python -W ignore src/run.py --model cmvaegnn --device cuda:0 --mode 15 --randomseed 1 --halfmixededge True
```

The reported graph-guided intervention encoders are selected with `--graphencoder True` and one of:

```text
--interv_encoder_type v1_trivalue
--interv_encoder_type v1_trivalue_subgraph1hop
--interv_encoder_type v2_dropedge
```

For the scFM feature experiment, pass an aligned `.h5ad` file through `--embedding_h5ad`; its cell embeddings are read from `obsm['embeddings']` by default. CMVAE-multihot retains the legacy preprocessing helper in `src/multihot.py` and expects precomputed 2,694-dimensional pathway features.

## Other Datasets

The other datasets in the paper are not path-only replacements. Reusing this code requires dataset-specific adapters for perturbation labels, train/validation/test splits, gene identifiers, feature dimensions, and compatible biological networks.

## Citation

```bibtex
@inproceedings{zhang2026gracevae,
  title={Causal Representation Learning from Network Data},
  author={Zhang, Jifan and Li, Michelle M. and Zheleva, Elena},
  booktitle={Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```
