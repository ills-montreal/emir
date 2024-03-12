import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")

# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    )
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--select", type=str, default="*")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")


    args = parser.parse_args()
    return args

# example of use of this script

# python EMIR/summarization/summarization_evaluation/evaluate_classification_task.py --model mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis --summaries data/summaries/rotten_tomatoes/rotten_tomatoes.csv --output output/rotten_tomatoes


def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file

    df = pd.read_csv(path, sep=";").dropna()

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df


def evaluate_classification_task(model, tokenizer, df, batch_size):
    # make a list of the tuples (text, summary)


    texts = df.text.tolist()
    summaries = df.summary.tolist()

    ds = list(zip(texts, summaries))

    eval_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    metrics = {"l2": [], "l1": [], "dot": [], "kl": [], "proba_of_error": []}

    for batch in tqdm(eval_loader):
        # evaluate source texts first:
        texts = batch[0]
        summaries = batch[1]

        # tokenize the texts
        texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        texts = {k: v.to(model.device) for k, v in texts.items()}

        # tokenize the summaries
        summaries = tokenizer(
            summaries, padding=True, truncation=True, return_tensors="pt"
        )
        summaries = {k: v.to(model.device) for k, v in summaries.items()}

        # evaluate the texts
        outputs = model(**texts)
        texts_logits = outputs.logits

        # evaluate the summaries
        outputs = model(**summaries)
        summaries_logits = outputs.logits

        # check if prediction would be the same
        preds_text = texts_logits.argmax(-1)
        preds_summaries = summaries_logits.argmax(-1)

        successes = (preds_text == preds_summaries).float().detach().cpu()

        # compute the metrics
        l2 = (summaries_logits - texts_logits).pow(2).sum(-1).detach().cpu()
        l1 = (summaries_logits - texts_logits).abs().sum(-1).detach().cpu()
        dot = (summaries_logits * texts_logits).sum(-1).detach().cpu()
        kl = (
            F.kl_div(
                F.log_softmax(summaries_logits, dim=-1),
                F.softmax(texts_logits, dim=-1),
                reduction="none",
                log_target=True,
            )
            .sum(-1)
            .detach()
            .cpu()
        )

        metrics["l2"].extend(l2.tolist())
        metrics["l1"].extend(l1.tolist())
        metrics["dot"].extend(dot.tolist())
        metrics["kl"].extend(kl.tolist())
        metrics["proba_of_error"].extend((1 - successes).tolist())

    # compute the mean of the metrics

    return metrics


def main():
    args = parse_args()

    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    df = parse_summaries(args.summaries)

    metrics = evaluate_classification_task(model, tokenizer, df, args.batch_size)

    # make a dataframe with the metric
    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics.add_prefix(f"{sanitize_model_name(args.model)}/")

    # merge the metrics with the summaries
    df = parse_summaries(args.summaries)
    df = pd.concat([df, df_metrics], axis=1)

    # save the dataframe
    df.to_csv(args.summaries, sep=";")


if __name__ == "__main__":
    main()
