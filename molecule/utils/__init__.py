import os
from typing import List, Optional
import datamol as dm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.nn.pool as tgp
import time

from .descriptors import DESCRIPTORS, CONTINUOUS_DESCRIPTORS
from .model_factory import ModelFactory
from .scattering_wavelet import get_scatt_from_path


class MolecularFeatureExtractor:
    def __init__(
        self,
        device: str = "cpu",
        length: int = 1024,
        dataset: str = "ClinTox",
        mds_dim: int = 0,
        normalize: bool = True,
    ):
        self.graph_input = None
        self.device = device

        self.length = length
        self.dataset = dataset
        self.mds_dim = mds_dim
        self.normalize = normalize
        pass

    def get_features(
        self,
        smiles: List[str],
        name: str,
        mols: Optional[List[dm.Mol]] = None,
        feature_type: str = "descriptor",
        path: str = "",
    ):
        device = self.device
        length = self.length
        dataset = self.dataset
        mds_dim = self.mds_dim
        normalize = self.normalize  # ONLY APPLIES TO MODELS

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
                    if os.path.exists(f"data/{dataset}/preprocessed.sdf"):
                        molecular_embedding = get_scatt_from_path(
                            f"data/{args.dataset}/preprocessed.sdf"
                        )
                        return molecular_embedding
                    else:
                        raise ValueError(f"File data/{dataset}_3d.sdf does not exist.")

            transformer_name = name.replace("/", "_")
            if mds_dim == 0 or transformer_name in CONTINUOUS_DESCRIPTORS:
                if os.path.exists(f"data/{dataset}/{transformer_name}_{length}.npy"):
                    molecular_embedding = np.load(
                        f"data/{dataset}/{transformer_name}_{length}.npy"
                    )
                    molecular_embedding = torch.tensor(
                        molecular_embedding,
                        device=device,
                    )
                    assert len(molecular_embedding) == len(
                        smiles
                    ), "The number of smiles and the number of embeddings are not the same."
                else:
                    from .molfeat import (
                        get_molfeat_descriptors,
                    )  # cannot install molfeat on cluster so move import here

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
            if os.path.exists(f"data/{dataset}/{name}.npy"):
                molecular_embedding = torch.tensor(
                    np.load(f"data/{dataset}/{name}.npy"), device=device
                )
                print(f"Loaded {name}.npy")
            else:
                print(f"Computing {name}.npy")
                molecular_embedding = ModelFactory(name)(
                    smiles,
                    mols=mols,
                    path=path,
                    transformer_name=name,
                    device=device,
                    dataset=dataset,
                )
                np.save(f"data/{dataset}/{name}.npy", molecular_embedding.cpu().numpy())

            if normalize:
                molecular_embedding = (
                    molecular_embedding - molecular_embedding.mean(dim=0)
                ) / (molecular_embedding.std(dim=0) + 1e-8)

            return molecular_embedding

        raise ValueError(f"Invalid transformer name: {transformer_name}")
