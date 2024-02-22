import os
from functools import partial
from typing import List, Tuple, Dict

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import datamol as dm

from emir.estimators import KNIFEEstimator, KNIFEArgs

from utils import get_features
from parser_mol import (
    generate_knife_config_from_args,
)
from models.model_paths import get_model_path
from utils import get_features

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_embedders(args: argparse.Namespace):
    MODELS = get_model_path(models=args.models)
    embeddings_fn = {}
    for model_name, model_path in MODELS.items():
        embeddings_fn[model_name] = partial(
            get_features,
            path=model_path,
            feature_type="model",
            name=model_name,
            device=args.device,
        )

    for method in args.descriptors:
        if not method in embeddings_fn:
            embeddings_fn[method] = partial(
                get_features,
                name=method,
                length=args.fp_length,
                feature_type="descriptor",
                device=args.device,
                mds_dim=args.mds_dim,
            )
    for model_name in args.models:
        if not model_name in embeddings_fn:
            embeddings_fn[model_name] = partial(
                get_features, name=model_name, feature_type="model", device=args.device
            )

    return embeddings_fn


def get_knife_preds(
    x1: callable,
    x2: callable,
    knife_config: KNIFEArgs = None,
    kernel_marg: Dict = None,
) -> Tuple[float, float, float, List[float]]:
    knife_estimator = KNIFEEstimator(
        knife_config, x1.shape[1], x2.shape[1], precomputed_marg_kernel=kernel_marg
    )  # Learn x2 from x1
    mi, m, c = knife_estimator.eval(x1.float(), x2.float(), record_loss=True)
    return (
        mi,
        m,
        c,
        torch.tensor(knife_estimator.recorded_loss, device="cpu"),
        torch.tensor(knife_estimator.recorded_marg_ent, device="cpu"),
        torch.tensor(knife_estimator.recorded_cond_ent, device="cpu"),
        knife_estimator.knife.kernel_marg,
    )


def get_knife_marg_kernel(
    emb_key: str,
    embeddings_fn: Dict[str, callable],
    knife_config: KNIFEArgs = None,
    smiles: List[str] = None,
    mols: List[dm.Mol] = None,
    dataset: str = None,
) -> Dict[str, torch.nn.Module]:
    x = embeddings_fn[emb_key](smiles, mols=mols, dataset=dataset).to(
        knife_config.device
    )
    if (x == 0).logical_or(x == 1).all():
        x = (x != 0).float()

    knife_estimator = KNIFEEstimator(
        knife_config, x.shape[1], x.shape[1]
    )  # Learn x2 from x1
    _ = knife_estimator.eval(x.float(), x.float(), record_loss=True, fit_only_marginal=True)
    return {emb_key: knife_estimator.knife.kernel_marg.to("cpu")}


def model_profile(
    model_name: str,
    args: argparse.Namespace,
    smiles: List[str],
    mols: List[dm.Mol] = None,
    marginal_kernels: Dict = {},
    p_bar: tqdm = None,
    embeddings_fn: Dict[str, callable] = None,
    knife_config: KNIFEArgs = None,
):
    results = {
        "X": [],
        "Y": [],
        "I(Y)": [],
        "I(Y|X)": [],
        "I(X->Y)": [],
        "I(X)": [],
        "I(X|Y)": [],
        "I(Y->X)": [],
        "Y_dim": [],
        "X_dim": [],
        "is_desc_discrete": [],
    }

    model_embedding = embeddings_fn[model_name](
        smiles, mols=mols, dataset=args.dataset
    ).to(knife_config.device)
    df_losses_XY = []
    df_losses_YX = []

    for desc in args.descriptors:
        mis = []
        descriptors_embedding = embeddings_fn[desc](
            smiles, mols=mols, dataset=args.dataset
        ).to(knife_config.device)
        if (
            len(descriptors_embedding.unique()) < 1500
            and not (descriptors_embedding == 0)
            .logical_or(descriptors_embedding == 1)
            .all()
        ):
            # setting to 1 all elements different from 0
            descriptors_embedding = (descriptors_embedding != 0).float()
        for i in range(args.n_runs):
            mi, m, c, loss, marg_ent, cond_ent, kernel_marg = get_knife_preds(
                descriptors_embedding,
                model_embedding,
                knife_config=knife_config,
                kernel_marg=marginal_kernels.get(model_name, None),
            )
            if model_name not in marginal_kernels:
                marginal_kernels[model_name] = kernel_marg

            results["Y"].append(model_name)
            results["X"].append(desc + str(args.fp_length))
            results["I(Y)"].append(m)
            results["I(Y|X)"].append(c)
            results["I(X->Y)"].append(mi)
            results["Y_dim"].append(model_embedding.shape[1])
            results["X_dim"].append(descriptors_embedding.shape[1])
            results["is_desc_discrete"].append(
                (descriptors_embedding == 0)
                .logical_or(descriptors_embedding == 1)
                .all()
                .item()
            )

            # saving the loss evolution in losses as csvs
            df_losses_XY.append(
                pd.DataFrame(
                    {
                        "loss": loss.cpu().numpy(),
                        "marg_ent": marg_ent.cpu().numpy(),
                        "cond_ent": cond_ent.cpu().numpy(),
                        "epoch": range(len(loss)),
                        "run": i,
                        "X": desc.replace("/", "_"),
                        "Y": model_name.replace("/", "_"),
                        "direction": "X->Y",
                    }
                )
            )
            if args.compute_both_mi:
                mi, m, c, loss, marg_ent, cond_ent, kernel_marg = get_knife_preds(
                    model_embedding,
                    descriptors_embedding,
                    knife_config=knife_config,
                    kernel_marg=marginal_kernels.get(desc, None),
                )
                del descriptors_embedding
                
                if desc not in marginal_kernels:
                    marginal_kernels[desc] = kernel_marg
                results["I(X)"].append(m)
                results["I(X|Y)"].append(c)
                results["I(Y->X)"].append(mi)
                df_losses_YX.append(
                    pd.DataFrame(
                        {
                            "loss": loss.cpu().numpy(),
                            "marg_ent": marg_ent.cpu().numpy(),
                            "cond_ent": cond_ent.cpu().numpy(),
                            "epoch": range(len(loss)),
                            "run": i,
                            "X": desc.replace("/", "_"),
                            "Y": model_name.replace("/", "_"),
                            "direction": "Y->X",
                        }
                    )
                )

            else:
                results["I(X)"].append(np.nan)
                results["I(X|Y)"].append(np.nan)
                results["I(Y->X)"].append(np.nan)

            if p_bar is not None:
                p_bar.update(1)
    del model_embedding

    if df_losses_XY != []:
        pd.concat(df_losses_XY).to_csv(
            os.path.join(
                os.path.join(args.out_dir, "losses"),
                f"{args.dataset}_{model_name.replace('/','_')}_{args.fp_length}_XY.csv",
            ),
            index=False,
        )
    if df_losses_YX != []:
        pd.concat(df_losses_YX).to_csv(
            os.path.join(
                os.path.join(args.out_dir, "losses"),
                f"{args.dataset}_{model_name.replace('/','_')}_{args.fp_length}_YX.csv",
            ),
            index=False,
        )
    results = pd.DataFrame(results)
    results.to_csv(
        os.path.join(
            args.out_dir,
            f"{args.dataset}_{model_name.replace('/', '_')}_{args.fp_length}.csv",
        ),
        index=False,
    )
    del df_losses_XY
    del df_losses_YX

    return results


def compute_all_mi(
    args: argparse.Namespace,
    smiles: List[str],
    mols: List[dm.Mol] = None,
    p_bar: tqdm = None,
) -> pd.DataFrame:
    from pathos.multiprocessing import ProcessingPool as Pool

    results = []
    marginal_kernels = {}

    knife_config = generate_knife_config_from_args(args)
    embeddings_fn = get_embedders(args)

    marginal_fn = partial(
        get_knife_marg_kernel,
        embeddings_fn=embeddings_fn,
        knife_config=knife_config,
        smiles=smiles,
        mols=mols,
        dataset=args.dataset,
    )
    all_embedders = args.descriptors + args.models
    p= Pool(args.n_jobs)
    all_marginal_kernels = p.map(marginal_fn, all_embedders)

    logger.info("All marginal kernels computed")
    for marginal_kernel in all_marginal_kernels:
        marginal_kernels.update(marginal_kernel)

    model_profile_partial = partial(
        model_profile,
        args=args,
        smiles=smiles,
        mols=mols,
        marginal_kernels=marginal_kernels,
        embeddings_fn=embeddings_fn,
        knife_config=knife_config,
    )
    p = Pool(args.n_jobs)
    results = p.map(model_profile_partial, args.models)
    return pd.concat(results)
