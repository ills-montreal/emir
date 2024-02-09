from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, RobertaModel

PIPELINE_CORRESPONDANCY = {
    "graphormer": "clefourrier/graphormer-base-pcqm4mv1",
    "MolBert": "jonghyunlee/MoleculeBERT_ChEMBL-pretrained",
    "ChemBertMLM-5M": "DeepChem/ChemBERTa-5M-MLM",
    "ChemBertMLM-10M": "DeepChem/ChemBERTa-10M-MLM",
    "ChemBertMLM-77M": "DeepChem/ChemBERTa-77M-MLM",
    "ChemBertMTR-5M": "DeepChem/ChemBERTa-5M-MTR",
    "ChemBertMTR-10M": "DeepChem/ChemBERTa-10M-MTR",
    "ChemBertMTR-77M": "DeepChem/ChemBERTa-77M-MTR",
}


def get_hugging_face_model(model_name):
    model_name = PIPELINE_CORRESPONDANCY.get(model_name, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name in ["DeepChem/ChemBERTa-5M-MTR"]:
        model = RobertaModel.from_pretrained(model_name)
    elif model_name in ["DeepChem/ChemBERTa-5M-MLM"]:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    return model, tokenizer
