import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


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


def eval_embeddings(model, df, batch_size):
    # make a list of the tuples (text, summary)

    texts = df.text.tolist()
    summaries = df.summary.tolist()

    ds = list(zip(texts, summaries))

    eval_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    metrics = {"l2": [], "l1": [], "dot": []}

    for batch in tqdm(eval_loader):
        # evaluate source texts first:
        texts = batch[0]
        summaries = batch[1]


        texts_emb = model.encode(texts, convert_to_tensor=True)
        summaries_emb = model.encode(summaries, convert_to_tensor=True)


        # compute the metrics
        l2 = (summaries_emb - texts_emb).pow(2).sum(-1).detach().cpu()
        l1 = (summaries_emb - texts_emb).abs().sum(-1).detach().cpu()
        dot = (summaries_emb * texts_emb).sum(-1).detach().cpu()

        metrics["l2"].extend(l2.tolist())
        metrics["l1"].extend(l1.tolist())
        metrics["dot"].extend(dot.tolist())

    # compute the mean of the metrics
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    return metrics


def main():
    args = parse_args()

    path = Path(args.output) / f"{args.summaries.stem}_metrics.csv"
    if args.select != "*":
        if args.select not in path.name:
            return


    model = SentenceTransformer(args.model)
    df = parse_summaries(args.summaries)

    metrics = eval_embeddings(model, df, args.batch_size)

    # make a dataframe with the metric
    df = pd.DataFrame(metrics, index=[args.summaries.stem])


    df = df.add_prefix(f"embeddings/{sanitize_model_name(args.model)}/")

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
