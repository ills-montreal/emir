import torch
import torch.nn as nn
from torchmdnet.models.model import load_model as load_model_torchmdnet
from molecule.external_repo.Frad.torchmdnet_frad.models.model import load_model as load_model_frad


name2path = {
    "DenoisingPretrainingPQCMv4": ("external_repo/pre-training-via-denoising/checkpoints/denoised-pcqm4mv2.ckpt", "torchmdnet"),
    "FRAD_QM9": ("backbone_pretrained_models/FRAD/frad.ckpt", "frad"),
}


class DenoisingModel(nn.Module):
    def __init__(self, name: str):
        super(DenoisingModel, self).__init__()
        path, module = name2path[name]
        load_fn = load_model_torchmdnet if module == "torchmdnet" else load_model_frad

        self.model = load_fn(path, derivative=False)

    def forward(self, z, pos, batch):
        x, v, z, pos, batch = self.model.representation_model(z, pos, batch=batch)
        return x, v, z, pos, batch


if __name__ == "__main__":
    model = DenoisingModel("DenoisingPretrainingPQCMv4")
    z = torch.tensor([1, 1, 8, 1, 1, 8], dtype=torch.long)
    pos = torch.rand(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    x, v, z, pos, batch = model(z, pos, batch)
    print(x)
