import os
import datamol as dm
from tqdm import tqdm
import argparse

from typing import List, Tuple
from molecule.utils.tdc_dataset import get_dataset
from utils.descriptors import can_be_2d_input

import pathos.multiprocessing as mp


def compute_3d(smiles: str):
    if "." in smiles:
        return None
    mol = dm.to_mol(smiles)
    if not can_be_2d_input(smiles, mol):
        return None
    desc = dm.descriptors.compute_many_descriptors(mol)
    if desc["mw"] > 1000:
        return None

    try:
        mol = dm.conformers.generate(
            dm.to_mol(smiles),
            align_conformers=True,
            ignore_failure=True,
            num_threads=8,
            n_confs=5,
        )
        return mol
    except Exception as e:
        print(e)
        return None


def precompute_3d(
    smiles: List[str],
    dataset_name: str = "tox21",
    n_jobs: int = 4,
    data_path: str = "data",
) -> Tuple[List[dm.Mol], List[str]]:
    if dataset_name.endswith(".csv"):
        dataset_name = dataset_name.replace(".csv", "")
    if os.path.exists(f"{data_path}/{dataset_name}_3d.sdf"):
        mols = dm.read_sdf(f"{data_path}/{dataset_name}_3d.sdf")
        smiles = [dm.to_smiles(m, True, False) for m in mols]
        return mols, smiles

    mols = []
    with mp.ProcessingPool(n_jobs) as pool:
        for mol in tqdm(pool.uimap(compute_3d, smiles), total=len(smiles)):
            if mol is not None:
                mols.append(mol)

    smiles = [dm.to_smiles(m, True, False) for m in mols]
    dm.to_sdf(mols, f"{data_path}/{dataset_name}_3d.sdf")

    return mols, smiles


parser = argparse.ArgumentParser(
    description="Compute 3d conformers for a given dataset, and save them in data/<dataset_name>_3d.sdf",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset",
    type=str,
    default="BindingDB_Kd",
    required=False,
    help="Dataset to use",
)
parser.add_argument(
    "--n-jobs",
    type=int,
    default=4,
    required=False,
    help="Number of jobs to use",
)
parser.add_argument("--data-path", type=str, default="data")

if __name__ == "__main__":
    args = parser.parse_args()
    DATA_PATH = args.data_path
    if args.dataset.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(f"{DATA_PATH}/{args.dataset}")
    else:
        df = get_dataset(args.dataset)

    keys = df.keys()
    if "Drug" in keys:
        smiles = df["Drug"].drop_duplicates().tolist()
    else:
        smiles = df["smiles"].drop_duplicates().tolist()
    mols = None
    _ = precompute_3d(smiles, args.dataset, data_path=DATA_PATH)
