import os
from typing import List, Optional
import datamol as dm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.nn.pool as tgp

from .molfeat import get_molfeat_descriptors
from .descriptors import DESCRIPTORS, CONTINUOUS_DESCRIPTORS
from .model_factory import ModelFactory
from .scattering_wavelet import get_scatt_from_path


def get_features(
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    feature_type: str = "descriptor",
    name: str = "",
    length: int = 1024,
    dataset: str = "tox21",
    path: str = "",
    device: str = "cpu",
    mds_dim: int = 0,
    normalize: bool = True,
):
    if feature_type == "descriptor":
        if name == "ScatteringWavelet":
            if os.path.exists(f"data/{dataset}/scattering_wavelet.npy"):
                molecular_embedding = torch.tensor(
                    np.load(f"data/{dataset}/scattering_wavelet.npy"), device=device
                )
                assert len(molecular_embedding) == len(
                    smiles
                ), "The number of smiles and the number of embeddings are not the same."
                return molecular_embedding
            else:
                if os.path.exists(f"data/{dataset}_3d.sdf"):
                    molecular_embedding = get_scatt_from_path(f"data/{dataset}_3d.sdf")
                    return molecular_embedding
                else:
                    raise ValueError(f"File data/{dataset}_3d.sdf does not exist.")

        transformer_name = name.replace("/", "_")
        if mds_dim == 0 or transformer_name in CONTINUOUS_DESCRIPTORS:
            if os.path.exists(f"data/{dataset}/{transformer_name}_{length}.npy"):
                molecular_embedding = torch.tensor(
                    np.load(f"data/{dataset}/{transformer_name}_{length}.npy"),
                    device=device,
                )
                assert len(molecular_embedding) == len(
                    smiles
                ), "The number of smiles and the number of embeddings are not the same."
            else:
                molecular_embedding = get_molfeat_descriptors(
                    smiles,
                    mols=mols,
                    transformer_name=transformer_name,
                    length=length,
                    dataset=dataset,
                )
            return molecular_embedding
        else:
            if os.path.exists(
                f"data/{dataset}/{transformer_name}_{length}_mds_{mds_dim}.npy"
            ):
                molecular_embedding = torch.tensor(
                    np.load(
                        f"data/{dataset}/{transformer_name}_{length}_mds_{mds_dim}.npy"
                    ),
                    device=device,
                )
                assert len(molecular_embedding) == len(
                    smiles
                ), "The number of smiles and the number of embeddings are not the same."

                # normalize
                molecular_embedding = (
                    molecular_embedding - molecular_embedding.mean()
                ) / (molecular_embedding.std() + 1e-8)

                return molecular_embedding
            else:
                raise ValueError(
                    f"File data/{dataset}/{transformer_name}_{length}_mds_{mds_dim}.npy does not exist."
                )

    if feature_type == "model":
        molecular_embedding = ModelFactory(name)(
            smiles,
            mols=mols,
            path=path,
            transformer_name=name,
            device=device,
        )

        if normalize:
            molecular_embedding = (
                molecular_embedding - molecular_embedding.mean(dim=0)
            ) / (molecular_embedding.std(dim=0) + 1e-8)

        return molecular_embedding

    raise ValueError(f"Invalid transformer name: {transformer_name}")
