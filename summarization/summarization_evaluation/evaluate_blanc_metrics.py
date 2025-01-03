import argparse
from pathlib import Path

import pandas as pd

from blanc import BlancHelp


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
    parser.add_argument("--summaries", type=Path, default="")

    # device
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output", type=str)

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


def evaluate_bartbert(df, device="cuda"):
    # make a list of the tuples (text, summary)

    texts = df.text.tolist()
    summaries = df.summary.tolist()

    scorer = BlancHelp(device="cuda", inference_batch_size=128, show_progress_bar=True)

    texts = [text.replace("\n", " ") for text in texts]
    summaries = [summary.replace("\n", " ") for summary in summaries]

    metrics = {"BLANC": []}
    metrics["BLANC"] = scorer.eval_pairs(texts, summaries)

    # compute the mean of the metrics
    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    return metrics


def main():
    args = parse_args()

    path = Path(args.output) / f"{args.summaries.stem}_metrics.csv"
    # create the output directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # check if the file exists already
    # if it does stop the execution
    if path.exists():
        raise ValueError("The file already exists.")

    # load the model
    df = parse_summaries(args.summaries)

    metrics = evaluate_bartbert(df)

    # make a dataframe with the metric
    df = pd.DataFrame(metrics, index=[args.summaries.stem])

    # Add the model name in the metrics names
    df = df.add_prefix(f"common/")

    # save the dataframe

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

    df.to_csv(path)


if __name__ == "__main__":
    main()
