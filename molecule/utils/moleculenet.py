import torch_geometric.nn.pool as tgp

from models.moleculenet_models import GNN
from typing import List, Optional
import datamol as dm
from torch.utils.data import DataLoader

import torch

MODEL_PARAMS = {
    "num_layer": 5,
    "emb_dim": 300,
    "JK": "last",
    "drop_ratio": 0.5,
    "gnn_type": "gin",
}


@torch.no_grad()
def get_embeddings_from_model_moleculenet(
    dataloader: DataLoader,
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    path: str = "backbone_pretrained_models/GROVER/grover.pth",
    pooling_method=tgp.global_mean_pool,
    normalize: bool = False,
    transformer_name: str = "",
):
    embeddings = []
    molecule_model = GNN(**MODEL_PARAMS)
    if not path == "":
        molecule_model.load_state_dict(torch.load(path))
    for b in dataloader:
        emb = pooling_method(molecule_model(b.x, b.edge_index, b.edge_attr), b.batch)
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=1)
        embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings
