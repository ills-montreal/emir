import torch
from typing import List
import numpy as np
import datamol as dm
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch_geometric.nn as nn
from tqdm import tqdm

from molecule.models.denoising_models import DenoisingModel

@torch.no_grad()
def get_embeddings_from_model_denoising(
    smiles: List[str],
    mols: List[dm.Mol],
    pooling_method="mean",
    normalize: bool = False,
    transformer_name: str = "",
    device: str = "cpu",
    batch_size: int = 16,
    **kwargs
):
    model = DenoisingModel(transformer_name).to(device)
    model.eval()
    n_molecules = len(mols)

    pos = []
    valence_charges = []

    for i, mol in enumerate(mols):
        for j, c in enumerate(mol.GetConformers()):
            pos.append(c.GetPositions())
            valence_charges.append([a.GetAtomicNum() for a in mol.GetAtoms()])
            break
    embeddings = []
    for i in tqdm(range(0, n_molecules, batch_size), desc="Computing embeddings"):
        batch_pos = pos[i : min(i + batch_size, n_molecules)]
        batch_valence_charges = valence_charges[i : min(i + batch_size, n_molecules)]
        batch = [[i] * len(b) for i, b in enumerate(batch_pos)]

        batch_pos = np.concatenate(batch_pos, axis=0)
        batch_valence_charges = np.concatenate(batch_valence_charges, axis=0)
        batch = np.concatenate(batch, axis=0)

        batch = torch.tensor(batch, device=device)
        batch_pos = torch.tensor(batch_pos, device=device)
        batch_valence_charges = torch.tensor(batch_valence_charges, device=device)

        x, _, _, _, _ = model(z = batch_valence_charges.long(), pos=batch_pos.float(), batch=batch.long())

        if pooling_method == "mean":
            x = nn.global_mean_pool(x, batch)
        elif pooling_method == "add":
            x = nn.global_add_pool(x, batch)
        elif pooling_method == "max":
            x = nn.global_max_pool(x, batch)
        else:
            raise ValueError(f"Pooling method {pooling_method} not recognized.")

        embeddings.append(x)

    return torch.cat(embeddings, dim=0)