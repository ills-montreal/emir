import torch
import torch.nn as nn

from molecule.external_repo.MolR.src.featurizer import MolEFeaturizer


class MolR(nn.Module):
    def __init__(self, name: str, batch_size: int = 256, device = "cpu", **kwargs):
        super(MolR, self).__init__()
        name = name.split("_")[1]
        path = f"backbone_pretrained_models/MolR/{name}_1024"
        self.model = MolEFeaturizer(path)
        self.batch_size = batch_size
        self.device = device

    def forward(self, smiles: str,):
        return torch.Tensor(self.model.transform(smiles, batch_size=self.batch_size)[0]).to(self.device)


if __name__ == "__main__":
    model = MolR("Molr_gat")
    smiles = ["CCO", "CCN", "CCO", "CCN", "jcii"]
    x = model(smiles)
    print(x)
