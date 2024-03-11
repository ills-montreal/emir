import os
import datamol as dm
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json

from utils import MolecularFeatureExtractor
from precompute_3d import precompute_3d
from utils.descriptors import DESCRIPTORS, can_be_2d_input
from utils.molfeat import get_molfeat_transformer


from tdc_dataset import get_dataset

parser = argparse.ArgumentParser(
    description="Compute ",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--dataset",
    type=str,
    default="LD50_Zhu",
    required=False,
    help="Dataset to use",
)

parser.add_argument(
    "--descriptors",
    type=str,
    nargs="+",
    default=DESCRIPTORS,
    required=False,
    help="List of descriptors to compute",
)


def main():
    args = parser.parse_args()
    if not os.path.exists(f"data/{args.dataset}/preprocessed.sdf"):
        if os.path.exists(f"data/{args.dataset}_3d.sdf"):
            print(f"Loading 3D conformers from data/{args.dataset}_3d.sdf")
            mols, smiles = precompute_3d(None, args.dataset)
        else:
            df = get_dataset(args.dataset)
            if "Drug" in df.columns:
                smiles = df["Drug"].tolist()
            else:
                smiles = df["smiles"].tolist()
            mols = None
            mols, smiles = precompute_3d(smiles, args.dataset)
        # Removing molecules that cannot be featurized
        for t_name in ["usr", "electroshape", "usrcat"]:
            transformer, thrD = get_molfeat_transformer(t_name)
            feat, valid_id = transformer(mols, ignore_errors=True)
            smiles = np.array(smiles)[valid_id]
            mols = np.array(mols)[valid_id]

        valid_smiles = []
        valid_mols = []
        for i, s in enumerate(tqdm(smiles, desc="Generating graphs")):
            if can_be_2d_input(s, mols[i]) and not "." in s:
                valid_smiles.append(s)
                valid_mols.append(mols[i])

        smiles = valid_smiles
        mols = valid_mols
        if not os.path.exists(f"data/{args.dataset}"):
            os.makedirs(f"data/{args.dataset}")

        pre_processed = pd.DataFrame({"smiles": smiles, "mols": mols})
        dm.to_sdf(
            pre_processed, f"data/{args.dataset}/preprocessed.sdf", mol_column="mols"
        )
        # save the SMILES in a json file
        with open(f"data/{args.dataset}/smiles.json", "w") as f:
            json.dump(smiles, f)

    else:
        pre_processed = dm.read_sdf(
            f"data/{args.dataset}/preprocessed.sdf", as_df=True, mol_column="mols"
        )
        smiles = pre_processed["smiles"].iloc[:, 0].tolist()
        mols = pre_processed["mols"].tolist()

    for desc in tqdm(args.descriptors, position=0, desc="Descriptors"):
        for length in tqdm(
            [1024], desc="Length", position=1, leave=False
        ):
            feature_extractor = MolecularFeatureExtractor(
                device="cpu",
                length=length,
                dataset=args.dataset,
                mds_dim=0,
            )
            if not os.path.exists(f"data/{args.dataset}/{desc}_{length}.npy"):
                descriptor = feature_extractor.get_features(
                    smiles, name=desc, mols=mols, feature_type="descriptor"
                ).numpy()

                np.save(
                    f"data/{args.dataset}/{desc.replace('/','_')}_{length}.npy",
                    descriptor,
                )
                del descriptor
            else:
                print(f"data/{args.dataset}/{desc}_{length}.npy already exists")


if __name__ == "__main__":
    main()
