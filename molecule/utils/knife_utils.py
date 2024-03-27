import os
from functools import partial
from itertools import product
from typing import List, Tuple, Dict

import pickle
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import datamol as dm

from emir.estimators import KNIFEEstimator, KNIFEArgs

from molecule.models.transformers_models import PIPELINE_CORRESPONDANCY
from parser_mol import (
    generate_knife_config_from_args,
)
from models.model_paths import get_model_path
from utils import MolecularFeatureExtractor
from molecule.models.denoising_models import name2path

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import wandb


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
    args: argparse.Namespace = None,
) -> Dict[str, torch.nn.Module]:

    if os.path.exists(os.path.join(args.out_dir, "marginal_{}.pkl".format(emb_key))):
        with open(os.path.join(args.out_dir, "marginal_{}.pkl".format(emb_key)), "rb") as f:
            marginal_kernel = pickle.load(f)
        return {emb_key: marginal_kernel}


    x = embeddings_fn[emb_key](smiles, mols=mols).to("cpu")
    if (x == 0).logical_or(x == 1).all():
        x = (x != 0).float()

    knife_estimator = KNIFEEstimator(
        knife_config, x.shape[1], x.shape[1]
    )  # Learn x2 from x1
    _ = knife_estimator.eval(
        x.float(), x.float(), record_loss=True, fit_only_marginal=True
    )
    marg_ent = torch.tensor(knife_estimator.recorded_marg_ent, device="cpu")
    df_run_marg_kernel = pd.DataFrame(
        {
            "marg_ent": marg_ent.cpu().numpy(),
            "epoch": np.linspace(0, args.n_epochs, len(marg_ent)),
            "run": 0,
            "X": emb_key.replace("/", "_"),
        }
    )
    df_run_marg_kernel.to_csv(
        os.path.join(
            os.path.join(args.out_dir, "losses"),
            f"{args.dataset}_{emb_key.replace('/','_')}_{args.fp_length}_marg.csv",
        ),
        index=False,
    )

    with open(os.path.join(args.out_dir, "marginal_{}.pkl".format(emb_key.replace('/','_'))), "wb") as f:
        pickle.dump(knife_estimator.knife.kernel_marg.to("cpu"), f)

    return {emb_key: knife_estimator.knife.kernel_marg.to("cpu")}


def model_profile(
    x_y: Tuple[str, str],
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
    model_name, desc = x_y
    model_embedding = embeddings_fn[model_name](smiles, mols=mols).to(
        knife_config.device
    ).to("cpu")
    df_losses_XY = []
    df_losses_YX = []

    mis = []
    descriptors_embedding = embeddings_fn[desc](smiles, mols=mols).to(
        knife_config.device
    ).to("cpu")

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
                    "cond_ent": cond_ent.cpu().numpy(),
                    "epoch": np.linspace(0, args.n_epochs, len(cond_ent)),
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

            if desc not in marginal_kernels:
                marginal_kernels[desc] = kernel_marg
            results["I(X)"].append(m)
            results["I(X|Y)"].append(c)
            results["I(Y->X)"].append(mi)
            df_losses_YX.append(
                pd.DataFrame(
                    {
                        "cond_ent": cond_ent.cpu().numpy(),
                        "epoch": np.linspace(0, args.n_epochs, len(cond_ent)),
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
    del descriptors_embedding
    del model_embedding

    if df_losses_XY != []:
        df_losses_XY = pd.concat(df_losses_XY)
    else:
        df_losses_XY = None
    if df_losses_YX != []:
        df_losses_YX = pd.concat(df_losses_YX)
    else:
        df_losses_YX = None
    results = pd.DataFrame(results)

    return (model_name, (results, df_losses_XY, df_losses_YX))


def compute_all_mi(
    args: argparse.Namespace,
    smiles: List[str],
    mols: List[dm.Mol] = None,
) -> pd.DataFrame:
    results = []
    marginal_kernels = {}

    knife_config = generate_knife_config_from_args(args)

    feature_extractor = MolecularFeatureExtractor(
        device=args.device,
        length=args.fp_length,
        dataset=args.dataset,
        use_vae=args.use_VAE_embs,
        vae_path=f"data/{args.dataset}/VAE/latent_dim_{args.vae_latent_dim}/n_layers_{args.vae_n_layers}/intermediate_dim_{args.vae_int_dim}",
    )

    embeddings_fn = get_embedders(
        list(set(args.descriptors + args.models)), feature_extractor
    )

    marginal_fn = partial(
        get_knife_marg_kernel,
        embeddings_fn=embeddings_fn,
        knife_config=knife_config,
        smiles=smiles,
        mols=mols,
        args=args,
    )

    all_embedders = args.descriptors + args.models
    all_embedders = list(set(all_embedders))

    all_marginal_kernels = list(
        tqdm(map(marginal_fn, all_embedders), total=len(all_embedders))
    )

    log_concatenated_tables_from_dir(
        os.path.join(args.out_dir, "losses"), "marginals", ["_marg.csv"]
    )
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
    all_combinaisons = list(product(args.models, args.descriptors))
    results = list(
        tqdm(
            map(model_profile_partial, all_combinaisons),
            total=len(all_combinaisons),
        )
    )

    # save all df by concatenating those with the same model_name
    concatenated_df = {}
    concatenated_df_XY = {}
    concatenated_df_YX = {}
    for model_name, (df, df_losses_XY, df_losses_YX) in results:
        if model_name not in concatenated_df:
            concatenated_df[model_name] = df
        else:
            concatenated_df[model_name] = pd.concat(
                [concatenated_df[model_name], df], axis=0
            )

        if df_losses_XY is not None:
            if model_name not in concatenated_df_XY:
                concatenated_df_XY[model_name] = df_losses_XY
            else:
                concatenated_df_XY[model_name] = pd.concat(
                    [concatenated_df_XY[model_name], df_losses_XY], axis=0
                )
        if df_losses_YX is not None:
            if model_name not in concatenated_df_YX:
                concatenated_df_YX[model_name] = df_losses_YX
            else:
                concatenated_df_YX[model_name] = pd.concat(
                    [concatenated_df_YX[model_name], df_losses_YX], axis=0
                )

    for model_name, df in concatenated_df.items():
        df.to_csv(
            os.path.join(
                args.out_dir,
                f"{args.dataset}_{model_name.replace('/', '_')}_{args.fp_length}.csv",
            ),
            index=False,
        )

    for model_name, df in concatenated_df_XY.items():
        if not df.empty:
            df.to_csv(
                os.path.join(
                    os.path.join(args.out_dir, "losses"),
                    f"{args.dataset}_{model_name.replace('/','_')}_{args.fp_length}_XY.csv",
                ),
                index=False,
            )
            ratio = df.shape[0] // 20000 + 1
            df = df[df.epoch % ratio == 0]
    for model_name, df in concatenated_df_YX.items():
        if not df.empty:
            df.to_csv(
                os.path.join(
                    os.path.join(args.out_dir, "losses"),
                    f"{args.dataset}_{model_name.replace('/','_')}_{args.fp_length}_YX.csv",
                ),
                index=False,
            )
            ratio = df.shape[0] // 20000 + 1
            df = df[df.epoch % ratio == 0]

    return pd.concat([r[0] for _, r in results])


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
