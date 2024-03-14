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

    parser.add_argument("--summaries", type=Path, default="")
    # evaluate gold summaries
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

    df = pd.read_csv(path, sep=";")

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df


def evaluate_seahorse_questions(df):
    map_questionnumber_to_question = {
        "question1": "SH/Comprehensible",
        "question2": "SH/Repetition",
        "question3": "SH/Grammar",
        "question4": "SH/Attribution",
        "question5": "SH/Main ideas",
        "question6": "SH/Conciseness",
    }

    metrics = {}

    for question in [
        "question1",
        "question2",
        "question3",
        "question4",
        "question5",
        "question6",
    ]:

        metrics[map_questionnumber_to_question[question]] = df[question].map({'Yes': 1, 'No': 0, "Unsure": 0.5}).mean()

    return metrics



def main():
    args = parse_args()

    df = parse_summaries(args.summaries)

    metrics = evaluate_seahorse_questions(df)

    # make a dataframe with the metric
    df = pd.DataFrame(metrics, index=[args.summaries.stem])

    path = Path(args.output) / f"{args.summaries.stem}_metrics.csv"
    # create the output directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # check if exists already, if it does load it and add the new columns

    if path.exists():
        df_old = pd.read_csv(path, index_col=0)

        # concat only the new columns
        df = pd.concat([df_old, df[df.columns.difference(df_old.columns)]], axis=1)

    df.to_csv(path)


if __name__ == "__main__":
    main()
