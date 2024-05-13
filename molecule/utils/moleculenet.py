import os

import torch_geometric.nn.pool as tgp
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from typing import List, Optional
import datamol as dm
import torch
from tqdm import tqdm

from molecule.models.moleculenet_models import GNN
from molecule.utils.moleculenet_encoding import mol_to_graph_data_obj_simple


MODEL_PARAMS = {
    "num_layer": 5,
    "emb_dim": 300,
    "JK": "last",
    "drop_ratio": 0.0,
    "gnn_type": "gin",
}

DATA_PATH = "/export/livia/datasets/datasets/public/molecule/data"
DATA_PATH = "data" if not os.path.exists(DATA_PATH) else DATA_PATH

@torch.no_grad()
def get_embeddings_from_model_moleculenet(
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    path: str = "backbone_pretrained_models/GROVER/grover.pth",
    pooling_method=tgp.global_mean_pool,
    normalize: bool = False,
    transformer_name: str = "",
    device: str = "cpu",
    batch_size: int = 2048,
    dataset: Optional[str] = None,
    **kwargs,
):
    graph_input_path = f"{DATA_PATH}/{dataset}/graph_input" if dataset is not None else None
    embeddings = []
    molecule_model = GNN(**MODEL_PARAMS).to(device)
    if not path == "":
        molecule_model.load_state_dict(torch.load(path))

    molecule_model.eval()

    if graph_input_path is None or not os.path.exists(graph_input_path):
        graph_input = []
        for s in tqdm(smiles, desc="Converting smiles to graph data object"):
            try:
                graph_input.append(
                    mol_to_graph_data_obj_simple(dm.to_mol(s), smiles=s).to(device)
                )
            except Exception as e:
                print(f"Failed to convert {s} to graph data object.")
                raise e
        dataset = InMemoryDataset.save(graph_input, graph_input_path)

    graph_input = InMemoryDataset()
    graph_input.load(graph_input_path)

    dataloader = DataLoader(
        graph_input,
        batch_size=batch_size,
        shuffle=False,
    )

    for b in tqdm(
        dataloader,
        desc="Computing embeddings from model",
        total=len(graph_input) // batch_size + 1,
    ):
        emb = pooling_method(molecule_model(b.x, b.edge_index, b.edge_attr), b.batch)
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=1)
        embeddings.append(emb)
    embeddings = torch.cat(embeddings, dim=0)
    del molecule_model
    return embeddings
