from transformers import pipeline
from transformers import AutoTokenizer, AutoModel

PIPELINE_CORRESPONDANCY = {
    "graphormer": "clefourrier/graphormer-base-pcqm4mv1",
    "MolBert": "jonghyunlee/MoleculeBERT_ChEMBL-pretrained",
}


def get_hugging_face_model(model_name):
    model_name = PIPELINE_CORRESPONDANCY.get(model_name, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer
