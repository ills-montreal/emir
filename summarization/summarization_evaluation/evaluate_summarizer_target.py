import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from emir.estimators import KNIFEEstimator, KNIFEArgs
import re


# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="paraphrase-MiniLM-L6-v2")
    parser.add_argument("--summaries", type=Path, default="", nargs="+")
    parser.add_argument("--output", type=str)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=200)

    # stoping criterion
    parser.add_argument(
        "--stopping_criterion", type=str, default="max_epochs"
    )  # "max_epochs" or "early_stopping"
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--average", type=str, default="var")
    parser.add_argument("--cov_diagonal", type=str, default="var")
    parser.add_argument("--cov_off_diagonal", type=str, default="")
    parser.add_argument("--optimize_mu", default=True, action="store_true")

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

    df = pd.read_csv(path).dropna()

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


def main():
    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    args = parse_args()

    model = SentenceTransformer(args.model)

    precomputed_marg_kernel = None
    for k, summaries_path in enumerate(args.summaries):
        df = parse_summaries(summaries_path)
        text_embeddings, summary_embeddings = embedd_sourcetexts_and_summaries(
            df, model
        )

        if precomputed_marg_kernel is None:
            knife_args = KNIFEArgs(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in KNIFEArgs.__annotations__
                }
            )
            estimator = KNIFEEstimator(
                knife_args,
                text_embeddings.shape[1],
                text_embeddings.shape[1],
                precomputed_marg_kernel=None,
            )

            estimator.eval(text_embeddings, text_embeddings, fit_only_marginal=True)

            precomputed_marg_kernel = estimator.knife.kernel_marg

        try:
            evaluate_summarizer(
                args,
                summaries_path,
                text_embeddings,
                summary_embeddings,
                args.device,
                precomputed_marg_kernel,
            )
        except Exception as e:
            print(e)
            continue


def evaluate_summarizer(
    args,
    summaries_path: Path,
    text_embeddings,
    summary_embeddings,
    device,
    precomputed_marg_kernel,
):
    path = Path(args.output) / f"{summaries_path.stem}_metrics.csv"
    if path.exists():
        raise ValueError("The file already exists.")

    knife_args = KNIFEArgs(
        **{k: v for k, v in vars(args).items() if k in KNIFEArgs.__annotations__}
    )

    torch.cuda.empty_cache()

    estimator = KNIFEEstimator(
        knife_args,
        text_embeddings.shape[1],
        summary_embeddings.shape[1],
        precomputed_marg_kernel=precomputed_marg_kernel,
    )

    mi, marg_ent, cond_ent = estimator.eval(summary_embeddings, text_embeddings)
    mi_rev, marg_ent_rev, cond_ent_rev = 0, 0, 0

    if re.match(r"M[1-9]{0,2}", summaries_path.stem):
        model_name = summaries_path.stem
        dataset_name, decoding_config, date = "SummEval", None, None
    else:
        metadata = summaries_path.stem.split("-_-")
        model_name, dataset_name, decoding_config, date = metadata

    # make a pandas dataframe, use summaries filename as index

    df = pd.DataFrame(
        {
            "filename": [summaries_path.stem],
            "metadata/Embedding model": [args.model],
            "metadata/Decoding config": [decoding_config],
            "metadata/Date": [date],
            "metadata/Model name": [model_name],
            "metadata/Dataset name": [dataset_name],
            "I(text -> summary)": [mi_rev],
            "H(summary)": [marg_ent_rev],
            "H(summary|text)": [cond_ent_rev],
            "I(summary -> text)": [mi],
            "H(text)": [marg_ent],
            "H(text|summary)": [cond_ent],
        },
    )

    df = df.set_index("filename")

    # save the dataframe

    # create the output directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_old = pd.read_csv(path, index_col=0)

        # concat only the new columns
        df = pd.concat([df_old, df[df.columns.difference(df_old.columns)]], axis=1)

    df.to_csv(path)


if __name__ == "__main__":
    main()
