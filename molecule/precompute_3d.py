import os
import datamol as dm
import numpy as np
from tqdm import tqdm
import argparse

from typing import List, Tuple, Optional
from tdc_dataset import get_dataset
from utils.descriptors import can_be_2d_input


def precompute_3d(smiles: List[str], dataset_name: str = "tox21"):
    if os.path.exists(f"data/{dataset_name}_3d.sdf"):
        mols = dm.read_sdf(f"data/{dataset_name}_3d.sdf")
        smiles = [dm.to_smiles(m, True, False) for m in mols]
        return mols, smiles

    mols = []
    for s in tqdm(smiles, desc="Generating conformers"):
        if can_be_2d_input(s, dm.to_mol(s)):
            try:
                mol = dm.conformers.generate(
                    dm.to_mol(s), align_conformers=True, ignore_failure=True, num_threads=4, n_confs=5
                )
                mols.append(mol)
            except Exception as e:
                print(e)
                pass


    dm.to_sdf(mols, f"data/{dataset_name}_3d.sdf")

    return mols, smiles


parser = argparse.ArgumentParser(
    description="Compute 3d conformers for a given dataset, and save them in data/<dataset_name>_3d.sdf",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset",
    type=str,
    default="HIV",
    required=False,
    help="Dataset to use",
)

if __name__ == "__main__":
    args = parser.parse_args()
    df = get_dataset(args.dataset)
    keys = df.keys()
    if "Drug" in keys:
        smiles = df["Drug"].tolist()
    else:
        smiles = df["smiles"].tolist()
    mols = None
    _ = precompute_3d(smiles, args.dataset)
