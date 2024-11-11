"""Universal wrapper for the KNIFE estimator, adaptations of these wrappers can be found in the domain specific scripts."""

import os
import argparse
from itertools import product
from typing import List, Tuple
import yaml

import pandas as pd
import torch
from tqdm import tqdm

from emir.estimators import KNIFEEstimator, KNIFEArgs


def get_config_cls_estimator(estimator_config, estimator_name):
    if estimator_name == "KNIFE":
        return KNIFEArgs(**estimator_config), KNIFEEstimator
    raise ValueError(f"Estimator {estimator_name} not found")


class EmbedderEvaluator:
    def __init__(self, args: argparse.Namespace):
        """
        Wrapper for the emebdding evaluation
        :param args: argparse.Namespace object, contains the configuration for
        the etimator class, whose name is referenced in args.estimator and the
        configuration file in args.estimator_config
        """
        self.args = args
        self.config, self.estimator_cls = get_config_cls_estimator(
            self.get_estimator_config(), args.estimator
        )

        # Metrics
        self.metrics = pd.DataFrame()

    def get_estimator_config(self):
        """
        Load the estimator configuration from the yaml file
        :return:
        """
        with open(self.args.estimator_config, "r") as f:
            estimator_args = yaml.safe_load(f)
        return estimator_args

    def init_estimator(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Initialize the estimator class
        :param X: Embeddings of the models simulating the other one
        :param Y: Embeddings of the model to be simulated
        :return:
        """
        return self.estimator_cls(
            self.config,
            X.shape[1],
            Y.shape[1],
        )

    def estimate_model_pair(
        self, pair: Tuple[Tuple[torch.Tensor, str], Tuple[torch.Tensor, str]]
    ):
        """
        Estimate the mutual information between two models.
        :param pair: Tuple of two tuples, each containing the embeddings of the models and their names
        :return:
        """
        X, X_name = pair[0]
        Y, Y_name = pair[1]

        estimator = self.init_estimator(X, Y)
        mi, m, cond_cent = estimator.eval(X.float(), Y.float())
        row = {
            "X": X_name,
            "Y": Y_name,
            "X_dim": X.shape[1],
            "Y_dim": Y.shape[1],
            "I(X->Y)": mi,
            "H(Y)": m,
            "H(Y|X)": cond_cent,
        }
        if self.metrics.shape[0] == 0:
            self.metrics = pd.DataFrame(columns=row.keys())
        self.metrics.loc[self.metrics.shape[0]] = row

    def __call__(
        self,
        X: List[Tuple[torch.Tensor, str]],
        Y: List[Tuple[torch.Tensor, str]],
    ):
        """
        Run the evaluation of the embeddings
        :param X: A list of tuples containing the embeddings of the models simulating the other ones and their names.
        :param Y: A list of tuples containing the embeddings of the models to be simulated and their names.
        :return:
        """
        all_combinaisons = list(product(X, Y))
        results = list(
            tqdm(
                map(self.estimate_model_pair, all_combinaisons),
                total=len(all_combinaisons),
            )
        )

        return self.metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, help="Estimator name", default="KNIFE")
    parser.add_argument(
        "--estimator_config",
        type=str,
        help="Path to the configuration file for the estimator",
        default=os.path.join(os.path.dirname(__file__), "knife.yaml"),
    )
    args = parser.parse_args()

    evaluator = EmbedderEvaluator(args)
    embs = []
    embs_randn = [torch.rand(500, 4) + torch.randn(1, 4) for _ in range(4)]
    for i in range(5):
        embeddings = torch.cat(
            [embs_randn[j] if j >= i else torch.zeros(500, 4) for j in range(4)],
            dim=1,
        )
        embs.append((embeddings, f"X_{i}"))

    X = embs
    Y = embs

    metrics = evaluator(X, Y)
    print(metrics.pivot_table(index="X", columns="Y", values="I(X->Y)"))
