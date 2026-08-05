# GraCE-VAE: Causal Representation Learning from Network Data

**Authors:** Jifan Zhang*, Michelle M. Li*, Elena Zheleva

**Paper:** [https://doi.org/10.1145/3770855.3819023](https://doi.org/10.1145/3770855.3819023)

## Overview

Causal disentanglement from soft interventions is identifiable under the assumptions of linear interventional faithfulness and availability of both observational and interventional data. Prior work has focused on unstructured observations without leveraging known relational context among measured entities. In many scientific applications, however, the measured variables come with an observed interaction network that provides structured context, such as protein-protein interactions and pathway-gene membership. We propose GraCE-VAE, a graph-aware causal discrepancy variational autoencoder that treats pathway-level information as an auxiliary view of the latent causal programs. The graph neural network encoder conditions on this auxiliary pathway view and the biological graph to improve amortized inference, while the causal decoder remains a latent SCM with soft interventions. Assuming samples are i.i.d. within each intervention regime, we show that GraCE-VAE inherits the identifiability guarantees of causal discrepancy VAEs and identifies the latent causal graph and intervention targets up to the standard equivalence class. Experiments on three CRISPR perturbation datasets demonstrate that leveraging structured biological context improves prediction of interventional outcomes, including unseen perturbation combinations.

## Installation and Setup

### :one: Download the Repo

First, clone the GitHub repository:

```
git clone https://github.com/michellemli/GraCE-VAE
cd GraCE-VAE
```

### :two: Set Up Environment

This codebase leverages Python, PyTorch, PyTorch Geometric, etc. To create an environment with all of the required packages, please ensure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed and then execute the commands:

```
conda env create -f environment.yml
conda activate GraCE_env
```

## Cite

```
@article{zhang2026gracevae,
  title={Causal Representation Learning from Network Data},
  author={Zhang, Jifan and Li, Michelle M and Zheleva, Elena},
  journal={KDD},
  year={2026}
}
```
