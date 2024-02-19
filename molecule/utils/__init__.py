import os
from typing import List, Optional
import datamol as dm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.nn.pool as tgp

from .molfeat import get_molfeat_descriptors
from .descriptors import DESCRIPTORS
from .model_factory import ModelFactory


def get_features(
    dataloader: DataLoader,
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    feature_type: str = "descriptor",
    name: str = "",
    length: int = 1024,
    dataset: str = "tox21",
    path: str = "",
    device: str = "cpu",
):
    if feature_type == "descriptor":
        transformer_name = name.replace("/", "_")
        if os.path.exists(f"data/{dataset}/{transformer_name}_{length}.npy"):
            molecular_embedding = torch.tensor(
                np.load(f"data/{dataset}/{transformer_name}_{length}.npy"), device=device
            )
            assert len(molecular_embedding) == len(
                smiles
            ), "The number of smiles and the number of embeddings are not the same."
            return molecular_embedding
        else:
            molecular_embedding = get_molfeat_descriptors(
                dataloader,
                smiles,
                mols=mols,
                transformer_name=transformer_name,
                length=length,
                dataset=dataset,
            )
            return molecular_embedding

    if feature_type == "model":
        molecular_embedding = ModelFactory(name)(
            dataloader,
            smiles,
            mols=mols,
            path=path,
            transformer_name=name,
            device=device,
        )
        return molecular_embedding

    raise ValueError(f"Invalid transformer name: {transformer_name}")
