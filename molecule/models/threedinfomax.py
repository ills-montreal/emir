import os

from typing import List
from dataclasses import dataclass, field
from inspect import signature
import argparse
import yaml
import re
import dgl

from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from molecule.external_repo.threeDInfomax.datasets.inference_dataset import (
    InferenceDataset,
)
from molecule.external_repo.threeDInfomax.models.pna import PNA
from molecule.external_repo.threeDInfomax.datasets.custom_collate import (
    graph_only_collate,
)

def load_model(args, data, device):
    model = globals()[args.model_type](
        avg_d=data.avg_degree if hasattr(data, "avg_degree") else 1,
        device=device,
        **args.model_parameters
    )
    if args.pretrain_checkpoint:
        # get arguments used during pretraining
        with open(
            os.path.join(
                os.path.dirname(args.pretrain_checkpoint),
                "train_arguments.yaml",
            ),
            "r",
        ) as arg_file:
            pretrain_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        pretrain_args = argparse.Namespace()
        pretrain_args.__dict__.update(pretrain_dict)
        checkpoint = torch.load(
            args.pretrain_checkpoint, map_location=device
        )
        # get all the weights that have something from 'args.transfer_layers' in their keys name
        # but only if they do not contain 'teacher' and remove 'student.' which we need for loading from BYOLWrapper
        weights_key = (
            "model3d_state_dict" if args.transfer_3d == True else "model_state_dict"
        )
        pretrained_gnn_dict = {
            re.sub("^gnn\.|^gnn2\.", "node_gnn.", k.replace("student.", "")): v
            for k, v in checkpoint[weights_key].items()
            if any(transfer_layer in k for transfer_layer in args.transfer_layers)
            and "teacher" not in k
            and not any(to_exclude in k for to_exclude in args.exclude_from_transfer)
        }
        model_state_dict = model.state_dict()
        model_state_dict.update(
            pretrained_gnn_dict
        )  # update the gnn layers with the pretrained weights
        model.load_state_dict(model_state_dict)
    return model


@dataclass
class Config:
    transfer_3d: bool = False

    @classmethod
    def from_kwargs(cls, **kwargs):
        # fetch the constructor's signature
        cls_fields = {field for field in signature(cls).parameters}

        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                native_args[name] = val
            else:
                new_args[name] = val

        ret = cls(**native_args)

        for new_name, new_val in new_args.items():
            setattr(ret, new_name, new_val)
        return ret


class ThreeDInfoMax(nn.Module):
    def __init__(
        self,
        smiles: List[str],
        path: str = "external_repo/threeDInfomax/configs_clean/tune_QM9_homo.yml",
        device: str = "cpu",
        batch_size: int = 2,
    ):
        super(ThreeDInfoMax, self).__init__()
        self.device = device
        test_data = InferenceDataset(smiles, device)
        self.generate_config(path)

        model = load_model(self.config, data=test_data, device=device)
        self.model = model.eval().to(device)
        self.test_loader = DataLoader(
            test_data, batch_size=batch_size, collate_fn=graph_only_collate
        )

    def generate_config(self, path):
        args = {}
        with open(path, "r") as path:
            config_dict = yaml.load(path, Loader=yaml.FullLoader)
        self.config = Config.from_kwargs(**config_dict)

    def get_hidden_states(self, batch):
        self.model.node_gnn(batch)
        readouts_to_cat = [
            dgl.readout_nodes(batch, "feat", op=aggr)
            for aggr in self.model.readout_aggregators
        ]
        readout = torch.cat(readouts_to_cat, dim=-1)
        return readout

    def featurize_data(self):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = batch.to(self.device)
                embeddings.append(self.get_hidden_states(batch))
        return torch.cat(embeddings, dim=0)


if __name__ == "__main__":
    model = ThreeDInfoMax(
        ["COO", "CCCCO"],
        "external_repo/threeDInfomax/configs_clean/tune_QM9_homo.yml",
        "cpu",
    )
    print(model.featurize_data().shape)
