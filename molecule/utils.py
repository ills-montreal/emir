from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

import datamol as dm
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D
import torch_geometric.nn.pool as tgp
from torch_geometric.data import DataLoader


import torch

from models.moleculenet_models import GNN, GNN_graphpred
from moleculenet_encoding import mol_to_graph_data_obj_simple

MODEL_PARAMS = {
    "num_layer": 5,
    "emb_dim": 300,
    "JK": "last",
    "drop_ratio": 0.5,
    "gnn_type": "gin",
}

threeD_method_fpvec = ["usrcat", "electroshape", "usr"]
# threeD_method_moleculetransf = ["cats3d",]
fpvec_method = [
    "ecfp-count",
    "ecfp",
    "estate",
    "erg",
    "rdkit",
    "topological",
    "avalon",
    "maccs",
]
moleculetransf_method = [
    "scaffoldkeys",
    "cats2d",
]
pharmac_method = ["cats", "default", "gobbi", "pmapper"]


@torch.no_grad()
def get_embeddings_from_model(
    dataloader: DataLoader,
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    path: str = "backbone_pretrained_models/GROVER/grover.pth",
    pooling_method=tgp.global_mean_pool,
    normalize: bool = True,
):
    embeddings = []
    molecule_model = GNN(**MODEL_PARAMS)
    molecule_model.load_state_dict(torch.load(path))
    for b in dataloader:
        embeddings.append(
            torch.nn.functional.normalize(
                pooling_method(molecule_model(b.x, b.edge_index, b.edge_attr), b.batch),
                dim=1,
            )
        )
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def get_molfeat_transformer(transformer_name: str):
    if transformer_name in fpvec_method or transformer_name in threeD_method_fpvec:
        if transformer_name in threeD_method_fpvec:
            return (
                FPVecTransformer(
                    kind=transformer_name,
                    dtype=float,
                ),
                True,
            )
        return (
            FPVecTransformer(kind=transformer_name, dtype=float, length=1024),
            False,
        )
    elif transformer_name in moleculetransf_method:
        return (
            MoleculeTransformer(featurizer=transformer_name, dtype=float),
            False,
        )
    elif transformer_name in pharmac_method:
        return (
            MoleculeTransformer(
                featurizer=Pharmacophore2D(factory=transformer_name),
                dtype=float,
            ),
            False,
        )
    else:
        raise ValueError(f"Invalid transformer name: {transformer_name}")


def get_molfeat_descriptors(
    dataloader: DataLoader,
    smiles: List[str],
    transformer_name: str,
    mols: Optional[List[dm.Mol]] = None,
    normalize: bool = True,
):
    """
    Returns a list of descriptors for a given smiles string, obtained by using Molfeat's FPVecTransformer.
    """
    transformer, threeD = get_molfeat_transformer(transformer_name)

    if threeD and mols is None:
        mols = [
            dm.conformers.generate(dm.to_mol(s), align_conformers=True, n_confs=5)
            for s in tqdm(smiles, desc="Generating conformers")
        ]
    if threeD:
        molecular_embeddings = torch.tensor(transformer(mols, progress=True))
    else:
        molecular_embeddings = torch.tensor(transformer(smiles, progress=True))
    if normalize:
        molecular_embeddings = torch.nn.functional.normalize(
            molecular_embeddings, dim=1
        )
    return molecular_embeddings
