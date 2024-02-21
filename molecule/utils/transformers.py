import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as tgp
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutputWithPoolingAndCrossAttentions
from typing import List, Optional
import datamol as dm

from molecule.models.transformers_models import get_hugging_face_model


@torch.no_grad()
def get_embeddings_from_transformers_batch(
    model: AutoModel, token: AutoTokenizer, batch: List[str], device: str
):
    input_tok = token(
        batch, padding=True, truncation=True, return_tensors="pt", max_length=128
    ).to(device)
    model_out = model(**input_tok)
    if isinstance(model_out, MaskedLMOutput):
        embeddings = model_out.logits[:, 0, :]
    elif isinstance(model_out, BaseModelOutputWithPoolingAndCrossAttentions):
        embeddings = model_out.pooler_output
    else:
        embeddings = model_out

    return embeddings


def get_embeddings_from_transformers(
    smiles: List[str],
    mols: Optional[List[dm.Mol]] = None,
    transformer_name: str = "graphormer",
    device: str = "cpu",
    batch_size: int = 64,
    **kwargs,
):
    model, token = get_hugging_face_model(transformer_name)
    model = model.to(device)
    model.eval()
    all_embeddings = torch.tensor([], device=device)
    for i_batch in range(0, len(smiles), batch_size):
        batch = smiles[i_batch : min(len(smiles), i_batch + batch_size)]
        all_embeddings = torch.cat(
            [
                all_embeddings,
                get_embeddings_from_transformers_batch(model, token, batch, device),
            ],
            dim=0,
        )
    print(len(smiles), all_embeddings.shape)
    return all_embeddings
