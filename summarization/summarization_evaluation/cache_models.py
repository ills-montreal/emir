# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

import pandas as pd


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


# slauw87/bart_summarisation
# flax-community/t5-base.csv-cnn-dm
# sshleifer/distilbart-cnn-6-6
# sshleifer/distilbart-cnn-12-3
# sshleifer/distilbart-cnn-12-6
# sshleifer/distill-pegasus-xsum-16-4
# sshleifer/distill-pegasus-cnn-16-4
# sshleifer/distill-pegasus-xsum-16-8
# sshleifer/distilbart-xsum-6-6
# sshleifer/distill-pegasus-xsum-12-12
# sshleifer/distilbart-xsum-12-1
# sshleifer/distilbart-cnn-12-6
# airKlizz/mt5-base.csv-wikinewssum-all-languages
models = list(set(
    [
        "csebuetnlp/mT5_multilingual_XLSum",
        "slauw87/bart_summarisation",
        "google/pegasus-cnn_dailymail",
        "sshleifer/distilbart-cnn-6-6",
        "sshleifer/distilbart-cnn-12-3",
        "sshleifer/distilbart-cnn-12-6",
        "sshleifer/distill-pegasus-xsum-16-4",
        "sshleifer/distill-pegasus-cnn-16-4",
        "sshleifer/distill-pegasus-xsum-16-8",
        "sshleifer/distilbart-xsum-6-6",
        "sshleifer/distill-pegasus-xsum-12-12",
        "sshleifer/distilbart-xsum-12-1",
        "sshleifer/distilbart-cnn-12-6",
        # "airKlizz/mt5-base.csv-wikinewssum-all-languages",
        "Falconsai/text_summarization",
        "Falconsai/medical_summarization",
        "google/pegasus-large",
        "google/pegasus-multi_news",
        "google/pegasus-arxiv",
        "facebook/bart-large-cnn",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]
))


french = list(set(
    [
        "csebuetnlp/mT5_multilingual_XLSum",
        "moussaKam/barthez-orangesum-abstract",
        "plguillou/t5-base-fr-sum-cnndm",
    ]
))

german = list(set(
    ["Shahm/bart-german", "Shahm/t5-small-german", "Einmalumdiewelt/PegasusXSUM_GNAD"]
))

spanish = list(set(
    [
        "josmunpen/mt5-small-spanish-summarization",
        "IIC/mt5-spanish-mlsum",
        "mrm8488/bert2bert_shared-spanish-finetuned-summarization",
        "eslamxm/mt5-base-finetuned-Spanish",
    ]
))


model_sizes = {"metadata/#params": [], "metadata/Model name": []}

for model_name in german + spanish + french + models:
    if "mistral" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model_sizes["metadata/#params"].append(model.num_parameters())
    model_sizes["metadata/Model name"].append(sanitize_model_name(model_name))


df = pd.DataFrame(model_sizes)

df.to_csv("output/model_sizes.csv", index=False)
