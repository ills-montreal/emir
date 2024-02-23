import torch_geometric.nn.pool as tgp
from torch_geometric.loader import DataLoader
from typing import List, Optional
import datamol as dm
import torch

from models.moleculenet_models import GNN
from moleculenet_encoding import mol_to_graph_data_obj_simple


MODEL_PARAMS = {
    "num_layer": 5,
    "emb_dim": 300,
    "JK": "last",
    "drop_ratio": 0.5,
    "gnn_type": "gin",
}


@torch.no_grad()
def get_embeddings_from_model_moleculenet(
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    path: str = "backbone_pretrained_models/GROVER/grover.pth",
    pooling_method=tgp.global_mean_pool,
    normalize: bool = False,
    transformer_name: str = "",
    device: str = "cpu",
):
    embeddings = []
    molecule_model = GNN(**MODEL_PARAMS).to(device)
    if not path == "":
        molecule_model.load_state_dict(torch.load(path))
    molecule_model.eval()

    graph_input = []
    for s in smiles:
        try:
            graph_input.append(mol_to_graph_data_obj_simple(dm.to_mol(s)).to(device))
        except Exception as e:
            print(f"Failed to convert {s} to graph data object.")
            raise e

    dataloader = DataLoader(
        graph_input,
        batch_size=32,
        shuffle=False,
    )

    for b in dataloader:
        emb = pooling_method(molecule_model(b.x, b.edge_index, b.edge_attr), b.batch)
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=1)
        embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0)
    del molecule_model
    return embeddings
