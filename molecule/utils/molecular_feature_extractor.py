import os
from typing import List, Optional
import datamol as dm
import numpy as np
import torch

from .descriptors import DESCRIPTORS, CONTINUOUS_DESCRIPTORS
from .model_factory import ModelFactory
from molecule.utils.embedder_utils.scattering_wavelet import get_scatt_from_path
from .molfeat import get_molfeat_descriptors


class MolecularFeatureExtractor:
    def __init__(
        self,
        device: str = "cpu",
        length: int = 1024,
        dataset: str = "ClinTox",
        normalize: bool = True,
        use_vae: bool = False,
        vae_path: str = "",
        data_dir: str = "data",
    ):
        self.graph_input = None
        self.device = device

        self.length = length
        self.dataset = dataset
        self.normalize = normalize
        self.use_vae = use_vae
        self.vae_path = vae_path
        self.data_dir = os.path.join(data_dir, dataset)
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
        normalize = self.normalize  # ONLY APPLIES TO MODELS

        if feature_type == "descriptor":
            transformer_name = name.replace("/", "_")
            if (not self.use_vae) or not os.path.exists(
                os.path.join(self.vae_path, f"{transformer_name}.npy")
            ):
                if self.use_vae:
                    print(
                        "Loading normal embeddings as VAE does not exist for {}, missing path : {}".format(
                            name, os.path.join(self.vae_path, f"{transformer_name}.npy")
                        )
                    )

                if name == "ScatteringWavelet":
                    if os.path.exists(f"{self.data_dir}/scattering_wavelet.npy"):
                        molecular_embedding = torch.tensor(
                            np.load(f"{self.data_dir}/scattering_wavelet.npy"),
                            device=device,
                        )
                        assert len(molecular_embedding) == len(
                            smiles
                        ), "The number of smiles and the number of embeddings are not the same."
                        return molecular_embedding
                    else:
                        raise ValueError(
                            f"File {self.data_dir}/scattering_wavelet.npy does not exist."
                        )

                if os.path.exists(f"{self.data_dir}/{transformer_name}_{length}.npy"):
                    molecular_embedding = np.load(
                        f"{self.data_dir}/{transformer_name}_{length}.npy"
                    )
                    molecular_embedding = torch.tensor(
                        molecular_embedding,
                        device=device,
                    )
                    assert len(molecular_embedding) == len(
                        smiles
                    ), "The number of smiles and the number of embeddings are not the same."
                else:
                    molecular_embedding = get_molfeat_descriptors(
                        smiles,
                        transformer_name,
                        mols=mols,
                        dataset=dataset,
                        length=length,
                    )
                    np.save(
                        f"{self.data_dir}/{transformer_name}_{length}.npy",
                        molecular_embedding.cpu().numpy(),
                    )

            else:
                molecular_embedding = torch.tensor(
                    np.load(os.path.join(self.vae_path, f"{transformer_name}.npy")),
                    device=device,
                )
                assert len(molecular_embedding) == len(
                    smiles
                ), "The number of smiles and the number of embeddings are not the same."
            return molecular_embedding

        if feature_type == "model":
            if os.path.exists(f"{self.data_dir}/{name}.npy"):
                molecular_embedding = torch.tensor(
                    np.load(f"{self.data_dir}/{name}.npy"), device=device
                )
            else:
                molecular_embedding = ModelFactory(name)(
                    smiles,
                    mols=mols,
                    path=path,
                    transformer_name=name,
                    device=device,
                    dataset=dataset,
                )
                np.save(
                    f"{self.data_dir}/{name}.npy", molecular_embedding.cpu().numpy()
                )

            if normalize:
                molecular_embedding = (
                    molecular_embedding - molecular_embedding.mean(dim=0)
                ) / (molecular_embedding.std(dim=0) + 1e-8)

            return molecular_embedding

        raise ValueError(f"Invalid transformer name: {name}")
