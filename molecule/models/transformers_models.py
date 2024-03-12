from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, RobertaModel

PIPELINE_CORRESPONDANCY = {
    "MolBert": "jonghyunlee/MoleculeBERT_ChEMBL-pretrained",
    "ChemBertMLM-5M": "DeepChem/ChemBERTa-5M-MLM",
    "ChemBertMLM-10M": "DeepChem/ChemBERTa-10M-MLM",
    "ChemBertMLM-77M": "DeepChem/ChemBERTa-77M-MLM",
    "ChemBertMTR-5M": "DeepChem/ChemBERTa-5M-MTR",
    "ChemBertMTR-10M": "DeepChem/ChemBERTa-10M-MTR",
    "ChemBertMTR-77M": "DeepChem/ChemBERTa-77M-MTR",
    "ChemGPT-1.2B": "ncfrey/ChemGPT-1.2B",
    "ChemGPT-19M": "ncfrey/ChemGPT-19M",
    "ChemGPT-4.7M": "ncfrey/ChemGPT-4.7M",
}


def get_hugging_face_model(model_name):
    model_name = PIPELINE_CORRESPONDANCY.get(model_name, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "MTR" in model_name:
        model = RobertaModel.from_pretrained(model_name)
    elif "MLM" in model_name:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    elif "ChemGPT" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    return model, tokenizer
