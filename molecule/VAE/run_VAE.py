import os
import argparse
import json
import pandas as pd
import numpy as np

from molecule.VAE.model import VariationalAutoencoder, VAEArgs
from molecule.utils import MolecularFeatureExtractor
import wandb


def get_parser():
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument(
        "--descriptors",
        type=str,
        nargs="+",
        default=[
            "ecfp",
            "estate",
            #"erg",
            "rdkit",
            "topological",
            "avalon",
            "maccs",
            "secfp",
            "fcfp",
            "scaffoldkeys",
            "cats",
            "gobbi",
            "pmapper",
            "cats/3D",
            "gobbi/3D",
            "pmapper/3D",
        ],
        help="List of descriptors to compare",
    )
    parser.add_argument("--dataset", type=str, default="HIV", help="Dataset to use")
    parser.add_argument("--out-dir", type=str, default="data", help="Output file")
    parser.add_argument("--length", type=int, default=1024, help="Input dimension")
    parser.add_argument(
        "--intermediate-dim", type=int, default=256, help="Intermediate dimension"
    )
    parser.add_argument("--latent-dims", type=int, default=256, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    wandb.config.update(args)
    feature_extractor = MolecularFeatureExtractor(
        length=args.length, dataset=args.dataset, device="cuda"
    )

    df_results = pd.DataFrame()
    with open(f"data/{args.dataset}/smiles.json", "r") as f:
        smiles = json.load(f)

    for desc in args.descriptors:

        X = feature_extractor.get_features(
            smiles,
            name=desc,
            feature_type="descriptor",
        )
        vae_config = VAEArgs(
            input_dim=X.shape[1],
            intermediate_dim=args.intermediate_dim,
            latent_dims=args.latent_dims,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print(f"Training VAE for {desc} -- input dim {vae_config.input_dim} -- latent dim {vae_config.latent_dims}")

        vae = VariationalAutoencoder(vae_config)
        vae.train(args.epochs, X)

        df_res = pd.DataFrame(
            {
                "loss": vae.loss,
                "epoch": range(len(vae.loss)),
            }
        )
        df_res["descriptor"] = desc
        df_results = pd.concat([df_results, df_res])

        embeddings = vae.get_embeddings(X).cpu().numpy()
        # save
        os.makedirs(f"{args.out_dir}/{args.dataset}/VAE", exist_ok=True)
        np.save(
            f"{args.out_dir}/{args.dataset}/VAE/{desc.replace('/','_')}_{vae_config.input_dim}_{vae_config.latent_dims}.npy",
            embeddings,
        )
        wandb.log(
            {"loss_{}".format(desc.replace("/", "_")): wandb.Table(dataframe=df_res)}
        )
    wandb.log({"losses": wandb.Table(dataframe=df_results)})


if __name__ == "__main__":
    wandb.init(project="Emir")
    main()
