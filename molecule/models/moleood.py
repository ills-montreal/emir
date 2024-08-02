import os

import json
import pickle
from typing import List
import torch
import torch.nn as nn

from molecule.external_repo.MoleOOD.OGB.modules.model import Framework
from molecule.external_repo.MoleOOD.OGB.evaluate import build_backend_from_config

from ogb.utils.mol import smiles2graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

DATA_PATH = "/export/livia/datasets/datasets/public/molecule/data"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data"

class MoleOOD(nn.Module):
    def __init__(
        self, name: str, device: str="cpu"
    ):  # name in the format MoleOOD_{architecture}
        super(MoleOOD, self).__init__()

        splitted = name.split("_")
        dataset = splitted[1]
        architecture = splitted[2]
        base_backend = (
            f"backbone_pretrained_models/MoleOOD/config/{architecture}_base_dp0.1.json"
        )
        sub_backend = f"backbone_pretrained_models/MoleOOD/config/GIN_sub_dp0.1.json"
        model_path = f"backbone_pretrained_models/MoleOOD/saved_model/{architecture}.pth"

        with open(base_backend) as Fin:
            base_backend_config = json.load(Fin)
        with open(sub_backend) as Fin:
            sub_backend_config = json.load(Fin)

        base_backend = build_backend_from_config(base_backend_config)
        sub_backend = build_backend_from_config(sub_backend_config)

        main_model = Framework(
            base_model=base_backend,
            sub_model=sub_backend,
            base_dim=base_backend_config["result_dim"],
            sub_dim=sub_backend_config["result_dim"],
            num_tasks=1,
            dropout=0,
        ).to(device)
        best_model_para = torch.load(model_path, map_location=device)
        main_model.load_state_dict(best_model_para["main"])
        main_model = main_model.eval()

        self.model = main_model
        self.device = device

    def forward(self, batch_sub, batched_data):
        batched_data = batched_data.to(self.device)
        batch_sub = [eval(x) for x in batch_sub]

        return self.model.get_molecule_feature(batch_sub, batched_data)

    def get_dataloader_from_dataset_name(self, dataset_name: str, batch_size: int = 4,data_dir: str = DATA_PATH):
        if not data_dir.endswith(dataset_name):
            data_dir = f"{data_dir}/{dataset_name}"

        with open(f"{data_dir}/smiles.json", "r") as f:
            smiles = json.load(f)
        with open(f"{data_dir}/moleood/substructures.pkl", "rb") as f:
            substructures = pickle.load(f)
        data = []
        for s in smiles:
            graph = smiles2graph(s)
            data.append(
                Data(
                    x=torch.tensor(graph["node_feat"]),
                    edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                    edge_attr=torch.tensor(graph["edge_feat"]),
                    num_nodes=graph["num_nodes"],
                    smiles= s
                )
            )
        substructures = [str(s) for s in substructures]
        return DataLoader(list(zip(substructures, data)), batch_size=batch_size, shuffle=False)



if __name__ == "__main__":
    model = MoleOOD("MoleOOD_OGB_GIN", "cpu")
    test_dataset = "DILI"

    dataloader = model.get_dataloader_from_dataset_name(test_dataset)

    for batch_sub, batch_data in dataloader:
        print(model(batch_sub, batch_data))
        break

