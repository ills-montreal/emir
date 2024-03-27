import os
import argparse
import json
import pandas as pd
import numpy as np
import torch

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
            "erg",
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
    parser.add_argument("--dataset", type=str, default="ClinTox", help="Dataset to use")
    parser.add_argument("--out-dir", type=str, default="data", help="Output file")
    parser.add_argument("--length", type=int, default=1024, help="Input dimension")
    parser.add_argument(
        "--intermediate-dim", type=int, default=512, help="Intermediate dimension"
    )
    parser.add_argument("--latent-dims", type=int, default=256, help="Latent dimension")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--n-layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--dropout-rate", type=float, default=0.2, help="Dropout rate")

    parser.add_argument("--pca", action="store_true", help="Plot PCA to the embeddings")
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

        if X.unique().shape[0] > 1000:
            print(f"Descriptor {desc} has more than 1000 unique values, skipping")
        vae_config = VAEArgs(
            input_dim=X.shape[1],
            n_layers=args.n_layers,
            intermediate_dim=args.intermediate_dim,
            latent_dims=args.latent_dims,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout_rate=args.dropout_rate,
        )
        print(
            f"Training VAE for {desc} -- input dim {vae_config.input_dim} -- latent dim {vae_config.latent_dims}"
        )

        vae = VariationalAutoencoder(vae_config)

        save_dir = f"{args.out_dir}/{args.dataset}/VAE/latent_dim_{args.latent_dims}/n_layers_{args.n_layers}/intermediate_dim_{args.intermediate_dim}"
        os.makedirs(save_dir, exist_ok=True)

        vae.train_model(args.epochs, X, save_dir=save_dir)

        df_res = pd.DataFrame(
            {
                "mse": vae.mse + vae.val_mse,
                "loss": vae.loss + vae.val_loss,
                "kl": vae.kl + vae.kl_val,
                "type": ["train"] * len(vae.loss) + ["val"] * len(vae.val_loss),
                "epoch": list(range(len(vae.loss))) + list(range(len(vae.val_loss))),
            }
        )
        df_res["descriptor"] = desc
        df_results = pd.concat([df_results, df_res])

        df_results = df_results.sort_values("epoch")
        wandb.log({"losses_table": wandb.Table(dataframe=df_results)})

        # Load saved model
        vae = VariationalAutoencoder(vae_config)
        vae.load_state_dict(torch.load(f"{save_dir}/model.pt"))

        embeddings = vae.get_embeddings(X)

        if args.pca:
            U, S, V = torch.pca_lowrank(embeddings, q=2)
            embeddings_pca = torch.matmul(embeddings, V[:, :2])[:10000]

            table_pca = wandb.Table(
                dataframe=pd.DataFrame(
                    embeddings_pca.cpu().numpy(),
                    columns=["PC1_{}".format(desc), "PC2_{}".format(desc)],
                )
            )
            wandb.log(
                {
                    "pca_plot/pca_{}".format(
                        desc.replace("/", "_")
                    ): wandb.plot.scatter(
                        table_pca,
                        "PC1_{}".format(desc),
                        "PC2_{}".format(desc),
                        title="PCA {}".format(desc),
                    )
                }
            )
        # save
        #np.save(
        #    f"{save_dir}/{desc.replace('/','_')}.npy",
        #    embeddings.cpu().numpy(),
        #)

    df_results = df_results.sort_values("epoch")
    wandb.log({"losses_table": wandb.Table(dataframe=df_results)})


if __name__ == "__main__":
    wandb.init(project="Emir_VAE")
    main()
