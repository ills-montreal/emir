

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

    args = parser.parse_args()
    return args


def parse_summaries(path : Path):
    # read csv file

    df = pd.read_csv(path)

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df


