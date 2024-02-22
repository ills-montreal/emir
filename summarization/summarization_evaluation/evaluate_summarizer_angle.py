import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from angle_emb import AnglE
from tqdm import tqdm

from emir.estimators import KNIFEEstimator, KNIFEArgs


# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="paraphrase-MiniLM-L6-v2")
    parser.add_argument("--summaries", type=Path, default="")
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


def embedd_sourcetexts_and_summaries(df, model, batch_size=32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param df: a pandas dataframe with at least the columns 'text' and 'summary'
    :param model: a sentence transformer model
    :return: a tuple of two torch tensors, one for the text embeddings and one for the summary embeddings
    """

    # embedd the text and the summary

    # split list of texts and summaries into batches
    texts = df.text.tolist()
    summaries = df.summary.tolist()

    texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    summaries = [summaries[i:i + batch_size] for i in range(0, len(summaries), batch_size)]

    # embedd the texts and summaries
    text_embeddings = []
    summary_embeddings = []

    for text_batch, summary_batch in tqdm(zip(texts, summaries)):
        text_embeddings.append(model.encode(text_batch, to_numpy=False).cpu())
        summary_embeddings.append(model.encode(summary_batch, to_numpy=False).cpu())

    text_embeddings = torch.cat(text_embeddings)
    summary_embeddings = torch.cat(summary_embeddings)

    return text_embeddings, summary_embeddings


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
    path = Path(args.output) / f"{args.summaries.stem}_metrics.csv"
    if path.exists():
        raise ValueError("The file already exists.")
    # load the model
    model = AnglE.from_pretrained(args.model, pooling_strategy='cls')
    model = model.cuda()

    # parse the summaries
    df = parse_summaries(args.summaries)

    # embedd the text and the summary
    text_embeddings, summary_embeddings = embedd_sourcetexts_and_summaries(df, model)

    # compute the mutual information text -> summary
    mi, marg_ent, cond_ent = compute_mi(
        text_embeddings, summary_embeddings, args.device, args
    )

    # compute reverse mi: summary -> text
    mi_rev, marg_ent_rev, cond_ent_rev = compute_mi(
        summary_embeddings, text_embeddings, args.device, args
    )

    # Retrive metadata from stem
    metadata = args.summaries.stem.split("-_-")
    model_name, dataset_name, decoding_config, date = metadata

    # make a pandas dataframe, use summaries filename as index


    df = pd.DataFrame(
        {
            "filename" : [args.summaries.stem],
            "metadata/Embedding model" : [args.model],
            "metadata/Decoding config" : [decoding_config],
            "metadata/Date" : [date],
            "metadata/Model name" : [model_name],
            "metadata/Dataset name" : [dataset_name],

            "I(text -> summary)": [mi],
            "H(summary)": [marg_ent],
            "H(summary|text)": [cond_ent],
            "I(summary -> text)": [mi_rev],
            "H(text)": [marg_ent_rev],
            "H(text|summary)": [cond_ent_rev],
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
