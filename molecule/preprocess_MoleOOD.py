from rdkit.Chem import BRICS
import argparse
import os
import pickle
import subprocess
from tqdm import tqdm
import json

from molecule.external_repo.MoleOOD.OGB.modules.ChemistryProcess import get_substructure


def get_result_dir(work_dir):
    result_dir = os.path.join(work_dir, "moleood")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocessing For Dataset")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "hERG",
        ],
        help="the datasets to preprocess",
    )
    parser.add_argument(
        "--timeout",
        default=120,
        type=int,
        help="maximal time to process a single molecule, count int seconds",
    )
    parser.add_argument(
        "--method",
        choices=["recap", "brics"],
        default="brics",
        help="the method to decompose the molecule, brics or recap",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Path to the data folder",
    )

    args = parser.parse_args()
    print(args)
    for dataset in args.datasets:
        data_path = os.path.join(args.data_path, dataset)
        if os.path.exists(f"{data_path}/moleood/substructures.pkl"):
            print(f"Dataset {dataset} already preprocessed, skipping")
            continue
        if os.path.exists(f"{data_path}/smiles.json"):
            result_dir = get_result_dir(data_path)
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            with open(f"{data_path}/smiles.json", "r") as f:
                smiles = json.load(f)

            file_name = (
                "substructures.pkl"
                if args.method == "brics"
                else "substructures_recap.pkl"
            )
            file_name = os.path.join(result_dir, file_name)
            substruct_list = []
            for idx, smile in enumerate(tqdm(smiles)):
                tx = get_substructure(smile=smile, decomp=args.method)
                substruct_list.append(tx)

            with open(file_name, "wb") as Fout:
                pickle.dump(substruct_list, Fout)
