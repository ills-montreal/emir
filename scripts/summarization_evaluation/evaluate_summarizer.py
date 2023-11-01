from typing import Tuple

import pandas as pd
import numpy as np
import torch
import argparse

from tqdm import tqdm
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from emir.estimators import KNIFEEstimator, KNIFEArgs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="paraphrase-MiniLM-L6-v2")
    parser.add_argument("--summaries", type=Path, default="")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--average", type=str, default="var")
    parser.add_argument("--cov_diagonal", type=str, default="var")
    parser.add_argument("--cov_off_diagonal", type=str, default="var")
    parser.add_argument("--optimize_mu", type=bool, default=False)
    parser.add_argument("--simu_params", type=list, default=["source_data", "target_data", "method", "optimize_mu"])
    parser.add_argument("--cond_modes", type=int, default=8)
    parser.add_argument("--marg_modes", type=int, default=8)
    parser.add_argument("--use_tanh", type=bool, default=True)
    parser.add_argument("--init_std", type=float, default=0.01)
    parser.add_argument("--ff_residual_connection", type=bool, default=False)
    parser.add_argument("--ff_activation", type=str, default="relu")
    parser.add_argument("--ff_layer_norm", type=bool, default=True)
    parser.add_argument("--ff_layers", type=int, default=2)

    args = parser.parse_args()
    return args


def parse_summaries(path : Path):
    '''
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    '''
    # read csv file

    df = pd.read_csv(path)

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df

def embedd_sourcetexts_and_summaries(df, model) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    :param df: a pandas dataframe with at least the columns 'text' and 'summary'
    :param model: a sentence transformer model
    :return: a tuple of two torch tensors, one for the text embeddings and one for the summary embeddings
    '''


    # embedd the text and the summary
    text_embeddings = model.encode(df.text.tolist(), convert_to_tensor=True)
    summary_embeddings = model.encode(df.summary.tolist(), convert_to_tensor=True)

    return text_embeddings, summary_embeddings

def compute_mi(text_embeddings, summary_embeddings, device, args) -> torch.Tensor:
    '''
    :param text_embeddings: a torch tensor of text embeddings
    :param summary_embeddings: a torch tensor of summary embeddings
    :param device: a string specifying the device to use
    :param args: a namespace object with the arguments for the KNIFE estimator
    :return: a torch tensor with the mutual information between the text and the summary
    '''

    # initialize the KNIFE estimator
    knife_estimator = KNIFEEstimator(KNIFEArgs(**vars(args)), text_embeddings.shape[1], summary_embeddings.shape[1])

    # compute the mutual information
    mi = knife_estimator.eval(text_embeddings.to(device), summary_embeddings.to(device))

    return mi

def main():
    args = parse_args()

    # load the model
    model = SentenceTransformer(args.model)

    # parse the summaries
    df = parse_summaries(args.summaries)

    # embedd the text and the summary
    text_embeddings, summary_embeddings = embedd_sourcetexts_and_summaries(df, model)

    # compute the mutual information
    mi = compute_mi(text_embeddings, summary_embeddings, args.device, args)

    print(mi)






