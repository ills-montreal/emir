import os
import datamol as dm
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json
import selfies as sf

from utils import MolecularFeatureExtractor
from precompute_3d import precompute_3d
from utils.descriptors import can_be_2d_input
from utils.molfeat import get_molfeat_transformer


from molecule.utils.tdc_dataset import get_dataset

parser = argparse.ArgumentParser(
    description="Compute ",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=[
        "hERG",
    ],
)

parser.add_argument(
    "--descriptors",
    type=str,
    nargs="+",
    default=["ecfp", "maccs"],
    required=False,
    help="List of descriptors to compute",
)

parser.add_argument(
    "--data-path",
    type=str,
    default="data",
    required=False,
    help="Path to the data folder",
)


def main():
    args = parser.parse_args()
    for dataset in args.datasets:
        data_path = os.path.join(args.data_path, dataset)

        if not os.path.exists(f"{data_path}/preprocessed.sdf"):
            if os.path.exists(f"{data_path}_3d.sdf"):
                print(f"Loading 3D conformers from data/{dataset}_3d.sdf")
                mols, smiles = precompute_3d(None, dataset)
            else:
                df = get_dataset(dataset.replace("__", " "))
                if "Drug" in df.columns:
                    smiles = df["Drug"].tolist()
                else:
                    smiles = df["smiles"].tolist()
                mols = None
                mols, smiles = precompute_3d(smiles, dataset)
            # Removing molecules that cannot be featurized
            for t_name in ["usr", "electroshape", "usrcat"]:
                transformer, thrD = get_molfeat_transformer(t_name)
                feat, valid_id = transformer(mols, ignore_errors=True)
                smiles = np.array(smiles)[valid_id]
                mols = np.array(mols)[valid_id]

            valid_smiles = []
            valid_mols = []
            for i, s in enumerate(tqdm(smiles, desc="Generating graphs")):
                mol = mols[i]
                # compute molecular weight and limit it under 1000
                desc = dm.descriptors.compute_many_descriptors(mol)
                if desc["mw"] > 1000:
                    continue
                try:
                    _ = sf.encoder(s)
                    if can_be_2d_input(s, mols[i]) and not "." in s:
                        valid_smiles.append(s)
                        valid_mols.append(mols[i])
                except Exception as e:
                    print(f"Error processing {s}: {e}")
                    continue

            smiles = valid_smiles
            mols = valid_mols
            if not os.path.exists(f"{data_path}"):
                os.makedirs(f"{data_path}")

            pre_processed = pd.DataFrame({"smiles": smiles, "mols": mols})
            dm.to_sdf(pre_processed, f"{data_path}/preprocessed.sdf", mol_column="mols")
            # save the SMILES in a json file
            with open(f"{data_path}/smiles.json", "w") as f:
                json.dump(smiles, f)

        else:
            pre_processed = dm.read_sdf(
                f"{data_path}/preprocessed.sdf", as_df=True, mol_column="mols"
            )
            smiles = pre_processed["smiles"].iloc[:, 0].tolist()
            mols = pre_processed["mols"].tolist()

        for desc in tqdm(args.descriptors, position=0, desc="Descriptors"):
            for length in tqdm([1024], desc="Length", position=1, leave=False):
                feature_extractor = MolecularFeatureExtractor(
                    device="cpu",
                    length=length,
                    dataset=dataset,
                    data_dir=args.data_path,
                )
                if not os.path.exists(f"{data_path}/{desc}_{length}.npy"):
                    descriptor = feature_extractor.get_features(
                        smiles, name=desc, mols=mols, feature_type="descriptor"
                    ).numpy()

                    np.save(
                        f"{data_path}/{desc.replace('/','_')}_{length}.npy",
                        descriptor,
                    )
                    del descriptor
                else:
                    print(f"{data_path}/{desc}_{length}.npy already exists")


if __name__ == "__main__":
    main()
