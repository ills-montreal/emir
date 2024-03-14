import logging
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from transformers import FlavaProcessor, FlavaModel

from emir.estimators import KNIFEEstimator, KNIFEArgs

from sys import stdout


# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/flava-full")
    parser.add_argument("--captions", type=Path, default="")
    parser.add_argument("--output", type=str)

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


def parse_captions_images(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file

    df = pd.read_csv(path, sep=";")

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["image_path", "caption"]]):
        raise ValueError("The csv file must have the columns 'image_path' and 'caption'.")

    return df


def embed_images_and_captions(df, model) -> Tuple[torch.Tensor, torch.Tensor]:
    # use flava full model to embedd the texts and images

    pass





def compute_mi(
        image_embeddings, caption_embeddings, device, args
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param image_embeddings: a torch tensor of text embeddings
    :param caption_embeddings: a torch tensor of summary embeddings
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
        KNIFEArgs(**args_dict), image_embeddings.shape[1], caption_embeddings.shape[1]
    )

    # compute the mutual information
    mi, marg_ent, cond_ent = knife_estimator.eval(
        image_embeddings.to(device), caption_embeddings.to(device)
    )

    return mi, marg_ent, cond_ent


)
