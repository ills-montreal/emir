import torch
from torch.utils.data import DataLoader
import torch_geometric.transforms as tgp
from transformers import AutoModel, AutoTokenizer, GPTNeoForCausalLM
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutputWithPoolingAndCrossAttentions
from typing import List, Optional
import datamol as dm
import selfies as sf

from molecule.models.transformers_models import get_hugging_face_model


@torch.no_grad()
def get_embeddings_from_transformers_batch(
    model: AutoModel, token: AutoTokenizer, batch: List[str], device: str
):

    if isinstance(model, GPTNeoForCausalLM): # ChemGPT using SELFIES
        batch = [sf.encoder(s) for s in batch]
        input_tok = token(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=128
        ).to(device)
        model_out = model(output_hidden_states=True, **input_tok)
        embeddings = (input_tok["attention_mask"].unsqueeze(2) * model_out.hidden_states[-1]).sum(dim=1) / input_tok["attention_mask"].sum(dim=1).unsqueeze(1)
    else:
        input_tok = token(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=128
        ).to(device)
        model_out = model(**input_tok)
        if model_out.hidden_states is not None and len(model_out.hidden_states) > 1:
            embeddings = model_out.hidden_states[0]
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
    return all_embeddings
