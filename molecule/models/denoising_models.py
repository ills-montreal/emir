import torch
import torch.nn as nn
from torchmdnet.models.model import load_model


name2path = {
    "DenoisingPretrainingPQCMv4": "external_repo/pre-training-via-denoising/checkpoints/denoised-pcqm4mv2.ckpt",
}


class DenoisingModel(nn.Module):
    def __init__(self, name: str):
        super(DenoisingModel, self).__init__()
        self.model = load_model(name2path[name], derivative=False)

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
