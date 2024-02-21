import os
from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm

import datamol as dm
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D, Pharmacophore3D
from torch_geometric.data import DataLoader

import torch



threeD_method_fpvec = ["usrcat", "electroshape", "usr"]
fpvec_method = [
    "ecfp-count",
    "ecfp",
    "estate",
    "erg",
    "rdkit",
    "topological",
    "avalon",
    "maccs",
    "atompair-count",
    "topological-count",
    "fcfp-count",
    "secfp",
    "pattern",
    "fcfp",
]
moleculetransf_method = ["scaffoldkeys", "mordred"]
pharmac_method = ["cats", "default", "gobbi", "pmapper"]


def get_molfeat_transformer(transformer_name: str, length: int = 1024):
    if transformer_name in fpvec_method or transformer_name in threeD_method_fpvec:
        if transformer_name in threeD_method_fpvec:
            return (
                FPVecTransformer(
                    kind=transformer_name,
                    dtype=float,
                    length=length,
                ),
                True,
            )
        return (
            FPVecTransformer(kind=transformer_name, dtype=float, length=length),
            False,
        )
    elif transformer_name in moleculetransf_method:
        return (
            MoleculeTransformer(
                featurizer=transformer_name, dtype=float, length=length
            ),
            False,
        )
    elif transformer_name in pharmac_method:
        return (
            MoleculeTransformer(
                featurizer=Pharmacophore2D(factory=transformer_name, length=length),
                dtype=float,
            ),
            False,
        )
    elif (
        transformer_name.endswith("3D")
        and transformer_name[:-3] in pharmac_method
        and not transformer_name[:-3] == "default"
    ):
        return (
            MoleculeTransformer(
                featurizer=Pharmacophore3D(
                    factory=transformer_name[:-3], length=length
                ),
                dtype=float,
            ),
            True,
        )
    else:
        raise ValueError(f"Invalid transformer name: {transformer_name}")


def physchem_descriptors(
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    length: int = 1024,
):
    mols = [dm.to_mol(s) for s in smiles]
    df_descriptors = dm.descriptors.batch_compute_many_descriptors(mols)
    df_descriptors = df_descriptors.fillna(0)
    df_descriptors = torch.tensor(df_descriptors.astype(np.float32).to_numpy())
    df_descriptors = (df_descriptors - df_descriptors.mean(dim=0, keepdim=True)) / (
        df_descriptors.std(dim=0, keepdim=True) + 1e-6
    )
    return df_descriptors


def get_molfeat_descriptors(
    smiles: List[str],
    transformer_name: str,
    mols: Optional[List[dm.Mol]] = None,
    dataset: str = "freesolv",
    length: int = 1024,
):
    """
    Returns a list of descriptors for a given smiles string, obtained by using Molfeat's FPVecTransformer.
    """
    if transformer_name == "physchem":
        molecular_embeddings = physchem_descriptors(
            smiles, mols=mols, length=length
        )
    else:
        transformer, threeD = get_molfeat_transformer(transformer_name, length=length)

        if threeD and (mols is None or mols == []):
            mols = [
                dm.conformers.generate(dm.to_mol(s), align_conformers=True, n_confs=5)
                for s in tqdm(smiles, desc="Generating conformers")
            ]
        if threeD:
            molecular_embeddings = torch.tensor(transformer(mols, progress=True))
        else:
            molecular_embeddings = torch.tensor(transformer(smiles, progress=True))
    return molecular_embeddings
