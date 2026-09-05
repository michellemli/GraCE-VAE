import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
import scipy.sparse as sp
import project_config


def _to_numpy_row(matrix_row):
    if sp.issparse(matrix_row):
        return matrix_row.toarray().reshape(-1)
    return np.asarray(matrix_row).reshape(-1)


def _load_aligned_obsm_embeddings(
    adata,
    embedding_h5ad,
    embedding_obsm_key,
    embedding_align_mode='auto',
):
    emb_adata = sc.read_h5ad(embedding_h5ad)
    if embedding_obsm_key not in emb_adata.obsm:
        raise KeyError(
            f"Embedding key '{embedding_obsm_key}' not found in {embedding_h5ad}. "
            f"Available keys: {list(emb_adata.obsm.keys())}"
        )

    if len(emb_adata.obs_names) != len(adata.obs_names):
        raise ValueError(
            "Embedding adata and Norman adata have different numbers of cells: "
            f"{len(emb_adata.obs_names)} vs {len(adata.obs_names)}"
        )

    adata_obs = np.asarray(adata.obs_names)
    embedding_obs = np.asarray(emb_adata.obs_names)

    if np.array_equal(embedding_obs, adata_obs):
        emb_matrix = np.asarray(emb_adata.obsm[embedding_obsm_key], dtype=np.float32)
        return emb_matrix

    if embedding_align_mode == 'row_order':
        emb_matrix = np.asarray(emb_adata.obsm[embedding_obsm_key], dtype=np.float32)
        return emb_matrix

    if embedding_align_mode == 'auto':
        emb_index = {name: i for i, name in enumerate(emb_adata.obs_names)}
        if all(name in emb_index for name in adata_obs):
            order = np.array([emb_index[name] for name in adata_obs], dtype=np.int64)
            emb_matrix = np.asarray(emb_adata.obsm[embedding_obsm_key][order], dtype=np.float32)
            return emb_matrix

        # Fallback: user-provided external embedding often preserves the exact row order
        # of Norman cells while dropping or rewriting obs_names.
        emb_matrix = np.asarray(emb_adata.obsm[embedding_obsm_key], dtype=np.float32)
        return emb_matrix

    raise ValueError(
        "Embedding adata obs_names do not match Norman adata obs_names. "
        "Use embedding_align_mode='row_order' if the row order is known to be identical."
    )

    return emb_matrix


# read the single cell perturbation dataset.
# map the target genes of each cell to a binary vector, using a target gene list "perturb_targets".
# "perturb_type" specifies whether the returned object contains single trarget-gene samples, double target-gene samples, or both.
class SCDataset(Dataset):
    def __init__(
        self,
        datafile=None,
        perturb_type='single',
        perturb_targets=None,
        embedding_h5ad=None,
        embedding_obsm_key='embeddings',
        embedding_align_mode='auto',
    ):
        super(Dataset, self).__init__()
        assert perturb_type in ['single', 'double', 'both'], 'perturb_type not supported!'
        self.datafile = str(datafile or project_config.SCDATA)

        adata = sc.read_h5ad(self.datafile)
        self.embedding_h5ad = embedding_h5ad
        self.embedding_obsm_key = embedding_obsm_key
        self.embedding_align_mode = embedding_align_mode
        self.embedding_dim = 0
        self.rand_ctrl_embeddings = None
        self.ptb_orig_indices = None
        self.rand_ctrl_orig_indices = None

        full_embeddings = None
        if embedding_h5ad is not None:
            full_embeddings = _load_aligned_obsm_embeddings(
                adata,
                embedding_h5ad,
                embedding_obsm_key,
                embedding_align_mode=embedding_align_mode,
            )
            self.embedding_dim = int(full_embeddings.shape[1])

        if perturb_targets is None:
            ptb_targets = list(set().union(*[set(i.split(',')) for i in adata.obs['guide_ids'].value_counts().index]))
            ptb_targets.remove('')
        else:
            ptb_targets = perturb_targets
        self.ptb_targets = ptb_targets

        ctrl_mask = (adata.obs['guide_ids'] == '').to_numpy()
        self.ctrl_orig_indices = np.where(ctrl_mask)[0]
        self.ctrl_samples = adata[ctrl_mask].X.copy()

        if perturb_type == 'single':
            ptb_mask = ((~adata.obs['guide_ids'].str.contains(',')) & (adata.obs['guide_ids']!='')).to_numpy()
            ptb_adata = adata[ptb_mask].copy()
            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
            self.ptb_orig_indices = np.where(ptb_mask)[0]
            del ptb_adata
        elif perturb_type == 'double':
            ptb_mask = adata.obs['guide_ids'].str.contains(',').to_numpy()
            ptb_adata = adata[ptb_mask].copy()
            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
            self.ptb_orig_indices = np.where(ptb_mask)[0]
            del ptb_adata     
        else:
            ptb_mask = (adata.obs['guide_ids']!='').to_numpy()
            ptb_adata = adata[ptb_mask].copy() 
            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
            self.ptb_orig_indices = np.where(ptb_mask)[0]
            del ptb_adata

        self.rand_ctrl_choice = np.random.choice(self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True)
        self.rand_ctrl_samples = self.ctrl_samples[self.rand_ctrl_choice]
        self.rand_ctrl_orig_indices = self.ctrl_orig_indices[self.rand_ctrl_choice]
        if full_embeddings is not None:
            self.rand_ctrl_embeddings = full_embeddings[self.rand_ctrl_orig_indices]
        del adata

    def __getitem__(self, item):
        x = torch.from_numpy(_to_numpy_row(self.rand_ctrl_samples[item])).double()
        y = torch.from_numpy(_to_numpy_row(self.ptb_samples[item])).double()
        c = torch.from_numpy(self.ptb_ids[item]).double()
        if self.embedding_h5ad is None:
            return x, y, c

        self._ensure_embeddings_loaded()
        e = torch.from_numpy(np.asarray(self.rand_ctrl_embeddings[item]).reshape(-1)).double()
        sample_idx = torch.tensor(int(self.rand_ctrl_orig_indices[item]), dtype=torch.long)
        return x, y, c, e, sample_idx
    
    def __len__(self):
        return self.ptb_samples.shape[0]

    def _ensure_embeddings_loaded(self):
        if self.embedding_h5ad is None or self.rand_ctrl_embeddings is not None:
            return

        adata = sc.read_h5ad(self.datafile)
        full_embeddings = _load_aligned_obsm_embeddings(
            adata,
            self.embedding_h5ad,
            self.embedding_obsm_key,
            embedding_align_mode=self.embedding_align_mode,
        )
        self.embedding_dim = int(full_embeddings.shape[1])
        self.rand_ctrl_embeddings = full_embeddings[self.rand_ctrl_orig_indices]
        del adata

    def __getstate__(self):
        state = self.__dict__.copy()
        # Keep pickled dataloaders small and force a safe reload by index when reused.
        state['rand_ctrl_embeddings'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault('embedding_h5ad', None)
        self.__dict__.setdefault('rand_ctrl_embeddings', None)


def map_ptb_features(all_ptb_targets, ptb_ids):
    ptb_features = []
    for id in ptb_ids:
        feature = np.zeros(all_ptb_targets.__len__())
        feature[[all_ptb_targets.index(i) for i in id.split(',')]] = 1
        ptb_features.append(feature)
    return np.vstack(ptb_features)
