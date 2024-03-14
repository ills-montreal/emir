import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from emir.estimators import KNIFEEstimator, KNIFEArgs


# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="paraphrase-MiniLM-L6-v2")
    parser.add_argument("--summaries", type=Path, default="")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=200)

    # stoping criterion
    parser.add_argument("--stopping_criterion", type=str, default="max_epochs")  # "max_epochs" or "early_stopping"
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--average", type=str, default="")
    parser.add_argument("--cov_diagonal", type=str, default="")
    parser.add_argument("--cov_off_diagonal", type=str, default="")
    parser.add_argument("--optimize_mu", default=False, action="store_true")

    parser.add_argument("--cond_modes", type=int, default=8)
    parser.add_argument("--marg_modes", type=int, default=8)
    parser.add_argument("--use_tanh", default=True, action="store_true")
    parser.add_argument("--init_std", type=float, default=0.01)
    parser.add_argument("--ff_residual_connection", type=bool, default=False)
    parser.add_argument("--ff_activation", type=str, default="relu")
    parser.add_argument("--ff_layer_norm", default=True, action="store_true")
    parser.add_argument("--ff_layers", type=int, default=2)

    # example of use of this script

    args = parser.parse_args()
    return args


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


def embedd_sourcetexts_and_summaries(df, model) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param df: a pandas dataframe with at least the columns 'text' and 'summary'
    :param model: a sentence transformer model
    :return: a tuple of two torch tensors, one for the text embeddings and one for the summary embeddings
    """

    # embedd the text and the summary

    text_embeddings = model.encode(df.text.tolist(), convert_to_tensor=True)
    summary_embeddings = model.encode(df.summary.tolist(), convert_to_tensor=True)

    return text_embeddings, summary_embeddings


def compute_mi(
        text_embeddings, summary_embeddings, device, args):
    """
    :param text_embeddings: a torch tensor of text embeddings
    :param summary_embeddings: a torch tensor of summary embeddings
    :param device: a string specifying the device to use
    :param args: a namespace object with the arguments for the KNIFE estimator
    :return: a torch tensor with the mutual information between the text and the summary
    """

    # clean args from argparse to fit KNIFEArgs: remove the arguments that are not in KNIFEArgs
    args_dict = vars(args)
    args_dict = {
        key: value
        for key, value in args_dict.items()
        if key in KNIFEArgs.__annotations__
    }

    # initialize the KNIFE estimator
    knife_estimator = KNIFEEstimator(
        KNIFEArgs(**args_dict), text_embeddings.shape[1], summary_embeddings.shape[1]
    )

    # compute the mutual information
    mi = knife_estimator.eval_per_sample(
        text_embeddings.to(device), summary_embeddings.to(device)
    )

    return mi

def main():
    args = parse_args()

    # load the model
    model = SentenceTransformer(args.model)

    # parse the summaries
    df = parse_summaries(args.summaries)

    # embedd the text and the summary
    text_embeddings, summary_embeddings = embedd_sourcetexts_and_summaries(df, model)

    # compute the mutual information text -> summary
    mi = compute_mi(
        text_embeddings, summary_embeddings, args.device, args
    )

    # compute reverse mi: summary -> text
    mi_rev = compute_mi(
        summary_embeddings, text_embeddings, args.device, args
    )

    print(mi.shape)

    # add columns to df
    df["pmi(text -> summary)"] = mi
    df["pmi(summary -> text)"] = mi_rev
    # save the df
    df.to_csv(args.summaries, sep=";", index=False)

if __name__ == "__main__":
    main()
