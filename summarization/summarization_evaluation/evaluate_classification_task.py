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
    # evaluate gold summaries
    parser.add_argument("--gold", default=False, action="store_true")
    parser.add_argument("--select", type=str, default="*")

    parser.add_argument("--output", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
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


def evaluate_classification_task(model, tokenizer, df, batch_size, gold=False):
    # make a list of the tuples (text, summary)

    if gold:
        texts = df.text.tolist()
        summaries = df.gold_summary.tolist()
    else:
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
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    return metrics


def main():
    args = parse_args()

    path = Path(args.output) / f"{args.summaries.stem}_metrics.csv"
    if args.select != "*":
        if args.select not in path.name:
            return

    # load the model
    if args.model =="jonaskoenig/topic_classification_04":
        model = AutoModelForSequenceClassification.from_pretrained(args.model, from_tf=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(args.device)

    if args.model == "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1":
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    df = parse_summaries(args.summaries)

    metrics = evaluate_classification_task(model, tokenizer, df, args.batch_size, args.gold)

    # make a dataframe with the metric
    df = pd.DataFrame(metrics, index=[args.summaries.stem])

    # Add the model name in the metrics names
    if args.gold:
        df = df.add_prefix(f"{sanitize_model_name(args.model)}/gold/")
    else:
        df = df.add_prefix(f"{sanitize_model_name(args.model)}/")

    # save the dataframe

    path = Path(args.output) / f"{args.summaries.stem}_metrics.csv"
    # create the output directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # check if exists already, if it does load it and add the new columns

    if path.exists():
        df_old = pd.read_csv(path, index_col=0)

        # create the colums if they do not exist
        for col in df.columns:
            if col not in df_old.columns:
                df_old[col] = float("nan")

        # add entry to the dataframe
        for col in df.columns:
            df_old.loc[args.summaries.stem, col] = df.loc[args.summaries.stem, col]

        df = df_old


    df.to_csv(path)


if __name__ == "__main__":
    main()
