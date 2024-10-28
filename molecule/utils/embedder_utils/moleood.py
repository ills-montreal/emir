from typing import List
import torch

from molecule.models.moleood import MoleOOD


@torch.no_grad()
def get_embeddings_from_moleood(
    smiles: List[str],
    transformer_name: str = "",
    batch_size: int = 2048,
    device: str = "cpu",
    dataset: str = "DILI",
    data_dir: str = "data",
    **kwargs,
):
    model = MoleOOD(transformer_name, device)
    dataloader = model.get_dataloader_from_dataset_name(
        dataset, batch_size, data_dir=data_dir
    )

    embeddings = []
    for batch_sub, batch_data in dataloader:
        embeddings.append(model(batch_sub, batch_data))

    return torch.cat(embeddings, dim=0)


if __name__ == "__main__":
    for arch in ["GCN", "GIN", "SAGE"]:
        x = get_embeddings_from_moleood([], transformer_name=f"MoleOOD_OGB_{arch}")
        print(x.shape)
