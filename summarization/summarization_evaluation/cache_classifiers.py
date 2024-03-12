# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


multilingual = [
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned",
    "citizenlab/distilbert-base-multilingual-cased-toxicity",
    "papluca/xlm-roberta-base-language-detection",
    "christinacdl/XLM_RoBERTa-Clickbait-Detection-new",
]

french = [
    "bardsai/finance-sentiment-fr-base",
    "cmarkea/distilcamembert-base-sentiment",
]


for model_name in french + multilingual:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
