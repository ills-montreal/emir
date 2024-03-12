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
    parser.add_argument("--summaries_1", type=Path, default="")
    parser.add_argument("--summaries_2", type=Path, default="")

    parser.add_argument("--output", type=str)
    # evaluate gold summaries

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=200)

    # stoping criterion
    parser.add_argument(
        "--stopping_criterion", type=str, default="max_epochs"
    )  # "max_epochs" or "early_stopping"
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


def embedd_sourcetexts_and_summaries(
    df_1, df_2, model
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param model: a sentence transformer model
    :return: a tuple of two torch tensors, one for the text embeddings and one for the summary embeddings
    """

    summary_1_embeddings = model.encode(df_1.summary.tolist(), convert_to_tensor=True)
    summary_2_embeddings = model.encode(df_2.summary.tolist(), convert_to_tensor=True)

    return summary_1_embeddings, summary_2_embeddings


def compute_mi(
    text_embeddings, summary_embeddings, device, args
) -> Tuple[float, float, float]:
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
    mi, marg_ent, cond_ent = knife_estimator.eval(
        text_embeddings.to(device), summary_embeddings.to(device)
    )

    return mi, marg_ent, cond_ent


def main():
    args = parse_args()
    # check if the file exists already
    # if it does stop the execution
    path = Path(args.output) / f"{args.summaries_1.stem}-_-{args.summaries_2.stem}-_-_cross_mi.csv"

    # load the model
    model = SentenceTransformer(args.model)

    # parse the summaries
    df_1 = parse_summaries(args.summaries_1)
    df_2 = parse_summaries(args.summaries_2)

    # embedd the text and the summary
    summary_embeddings_1, summary_embeddings_2 = embedd_sourcetexts_and_summaries(
        df_1=df_1, df_2=df_2, model=model
    )

    # compute the mutual information 2 -> 1
    mi, marg_ent, cond_ent = compute_mi(
        summary_embeddings_2, summary_embeddings_1, args.device, args
    )

    # Retrive metadata from stem
    metadata = args.summaries_1.stem.split("-_-")
    model_name_1, dataset_name_1, decoding_config_1, date_1 = metadata

    metadata = args.summaries_2.stem.split("-_-")
    model_name_2, dataset_name_2, decoding_config_2, date_2 = metadata

    # make a pandas dataframe, use summaries filename as index

    df = pd.DataFrame(
        {
            "summary_1": [args.summaries_1.stem],
            "summary_2": [args.summaries_2.stem],
            "metadata/Embedding model": [args.model],

            "metadata/Decoding config 1": [decoding_config_1],
            "metadata/Date 1 ": [date_1],
            "metadata/Model name 1": [model_name_1],
            "metadata/Dataset name 1": [dataset_name_1],

            "metadata/Decoding config 2": [decoding_config_2],
            "metadata/Date 2": [date_2],
            "metadata/Model name 2": [model_name_2],

            "I(summary_1 -> summary_2)": [mi],
            "H(summary_2)": [marg_ent],
            "H(summary_2|summary_1)": [cond_ent],
        },
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path)


if __name__ == "__main__":
    main()
