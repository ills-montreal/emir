import os
from functools import partial
from itertools import product
from typing import List, Dict

import pickle
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import datamol as dm

from emir.estimators import KNIFEEstimator, KNIFEArgs
from emir.embedder_evaluator import get_config_cls_estimator

from molecule.models.transformers_models import PIPELINE_CORRESPONDANCY
from molecule.models.model_paths import get_model_path
from molecule.utils import MolecularFeatureExtractor
from molecule.models.denoising_models import name2path

import yaml
import logging
import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EstimatorRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        smiles: List[str],
        mols: List[dm.Mol] = None,
        p_bar=None,
    ):
        self.args = args
        self.config, self.estimator_cls = get_config_cls_estimator(
            self.get_estimator_config(), args.estimator
        )
        self.feature_extractor = MolecularFeatureExtractor(
            device=self.config.device,
            dataset=args.dataset,
            data_dir=args.data_path,
        )
        self.embeddings_fn = get_embedders(
            list(set(args.Y + args.X)), self.feature_extractor
        )
        self.smiles = smiles
        self.mols = mols

        self.p_bar = p_bar

        # Metrics
        self.metrics = pd.DataFrame()
        self.loss = pd.DataFrame()

    def get_estimator_config(self):
        with open(self.args.estimator_config, "r") as f:
            estimator_args = yaml.safe_load(f)
        return estimator_args

    def pre_run(self):
        pass

    def init_estimator(self, X, Y, Y_name):
        return self.estimator_cls(
            self.config,
            X.shape[1],
            Y.shape[1],
        )

    def estimate_model_pair(self, pair):
        X_name, Y_name = pair
        X = self.embeddings_fn[X_name](self.smiles, mols=self.mols).to("cpu")
        Y = self.embeddings_fn[Y_name](self.smiles, mols=self.mols).to("cpu")

        for i in range(self.args.n_runs):
            estimator = self.init_estimator(X, Y, Y_name)
            mi, m, cond_cent = estimator.eval(X.float(), Y.float())
            metrics = {"I(X->Y)": mi, "H(Y)": m, "H(Y|X)": cond_cent}
            loss_metrics = dict(
                loss=torch.tensor(estimator.recorded_loss, device="cpu"),
            )

            if self.p_bar is not None:
                self.p_bar.update(1)
            row = {
                "X": X_name,
                "Y": Y_name,
                "run": i,
                "X_dim": X.shape[1],
                "Y_dim": Y.shape[1],
            }
            for k, v in metrics.items():
                row[k] = v
            if self.metrics.shape[0] == 0:
                self.metrics = pd.DataFrame(columns=row.keys())
            self.metrics.loc[self.metrics.shape[0]] = row

            for i in range(self.config.n_epochs):
                row = {
                    "X": X_name,
                    "Y": Y_name,
                    "run": i,
                    "X_dim": X.shape[1],
                    "Y_dim": Y.shape[1],
                    "epoch": i,
                }
                for k, v in loss_metrics.items():
                    print(k, v.shape)
                    row[k] = v[i]
                if self.loss.shape[0] == 0:
                    self.loss = pd.DataFrame(columns=row.keys())
                self.loss.loc[self.loss.shape[0]] = row

    def __call__(
        self,
    ):
        self.pre_run()

        all_combinaisons = list(product(self.args.X, self.args.Y))
        results = list(
            tqdm(
                map(self.estimate_model_pair, all_combinaisons),
                total=len(all_combinaisons),
            )
        )


class MI_EstimatorRunner(EstimatorRunner):
    def __init__(self, args: argparse.Namespace, smiles: List[str], mols: List[dm.Mol]):
        super().__init__(args, smiles, mols)
        self.marginal_kernels = {}

    def pre_run(self):
        marginal_fn = partial(
            get_knife_marg_kernel,
            embeddings_fn=self.embeddings_fn,
            knife_config=self.config,
            smiles=self.smiles,
            mols=self.mols,
            args=self.args,
        )

        all_embedders = list(set(self.args.Y + self.args.X))
        np.random.shuffle(all_embedders)

        all_marginal_kernels = list(
            tqdm(map(marginal_fn, all_embedders), total=len(all_embedders))
        )
        if self.args.wandb:
            log_concatenated_tables_from_dir(
                os.path.join(self.args.out_dir, "losses"), "marginals", ["_marg.csv"]
            )
        logger.info("All marginal kernels computed")
        for marginal_kernel in all_marginal_kernels:
            self.marginal_kernels.update(marginal_kernel)

    def init_estimator(self, X, Y, Y_name):
        return self.estimator_cls(
            self.config,
            X.shape[1],
            Y.shape[1],
            precomputed_marg_kernel=self.marginal_kernels[Y_name],
        )


def get_embedders(all_embedders, feature_extractor):
    MODELS = get_model_path(models=all_embedders)
    embeddings_fn = {}

    for model_name, model_path in MODELS.items():
        embeddings_fn[model_name] = partial(
            feature_extractor.get_features,
            path=model_path,
            feature_type="model",
            name=model_name,
        )
    for model_name in all_embedders:
        if not model_name in embeddings_fn and (
            model_name in PIPELINE_CORRESPONDANCY.keys()
            or model_name in name2path
            or model_name.startswith("MolR")
            or model_name.startswith("MoleOOD")
            or model_name.startswith("ThreeDInfomax")
        ):
            embeddings_fn[model_name] = partial(
                feature_extractor.get_features,
                name=model_name,
                feature_type="model",
            )
    for method in all_embedders:
        if not method in embeddings_fn:
            embeddings_fn[method] = partial(
                feature_extractor.get_features,
                name=method,
                feature_type="descriptor",
            )

    return embeddings_fn


def get_knife_marg_kernel(
    emb_key: str,
    embeddings_fn: Dict[str, callable],
    knife_config: KNIFEArgs = None,
    smiles: List[str] = None,
    mols: List[dm.Mol] = None,
    args: argparse.Namespace = None,
) -> Dict[str, torch.nn.Module]:
    if os.path.exists(os.path.join(args.out_dir, "marginal_{}.pkl".format(emb_key))):
        logger.info(f"Loading marginal kernel for {emb_key}")
        with open(
            os.path.join(args.out_dir, "marginal_{}.pkl".format(emb_key)), "rb"
        ) as f:
            marginal_kernel = pickle.load(f)
        return {emb_key: marginal_kernel}

    x = embeddings_fn[emb_key](smiles, mols=mols).to("cpu")

    knife_estimator = KNIFEEstimator(
        knife_config, x.shape[1], x.shape[1]
    )  # Learn x2 from x1
    _ = knife_estimator.eval(x.float(), x.float(), fit_only_marginal=True)
    marg_ent = torch.tensor(knife_estimator.recorded_marg_ent, device="cpu")

    with open(
        os.path.join(args.out_dir, "marginal_{}.pkl".format(emb_key.replace("/", "_"))),
        "wb",
    ) as f:
        pickle.dump(knife_estimator.knife.kernel_marg.to("cpu"), f)

    return {emb_key: knife_estimator.knife.kernel_marg.to("cpu")}


def log_concatenated_tables_from_dir(path: str, name, template=["_marg.csv"]):
    all_dfs = []
    for file in os.listdir(path):
        for t in template:
            if file.endswith(t):
                all_dfs.append(pd.read_csv(os.path.join(path, file)))
    df = pd.concat(all_dfs)
    if df.shape[0] > 20000:
        ratio = df.shape[0] // 20000 + 1
        df = df[df.epoch % ratio == 0]
    wandb.log({name: wandb.Table(dataframe=df)})
    print(f"Logged {name} table")
