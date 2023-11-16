import os
import datamol as dm
import numpy as np
import pandas as pd
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list
from tqdm import tqdm
import argparse
import json
from typing import List, Tuple, Optional

from utils import get_features
from moleculenet_encoding import mol_to_graph_data_obj_simple
from precompute_3d import precompute_3d
from utils.descriptors import DESCRIPTORS
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





def main():
    args = parser.parse_args()
    if not os.path.exists(f"data/{args.dataset}/preprocessed.sdf"):
        df = get_dataset(args.dataset)

        smiles = df["Drug"].tolist()
        mols = None

        mols, smiles = precompute_3d(smiles, args.dataset)
        # Removing molecules that cannot be featurized
        for t_name in ["usr", "electroshape", "usrcat"]:
            transformer, thrD = get_molfeat_transformer(t_name)
            feat, valid_id = transformer(mols, ignore_errors=True)
            smiles = np.array(smiles)[valid_id]
            mols = np.array(mols)[valid_id]

        graph_input = []
        valid_smiles = []
        valid_mols = []
        for i, s in enumerate(tqdm(smiles, desc="Generating graphs")):
            try:
                graph_input.append(mol_to_graph_data_obj_simple(dm.to_mol(s)))
                valid_smiles.append(s)
                valid_mols.append(mols[i])
            except:
                pass
        smiles = valid_smiles
        mols = valid_mols
        if not os.path.exists(f"data/{args.dataset}"):
            os.makedirs(f"data/{args.dataset}")

        pre_processed = pd.DataFrame({"smiles": smiles, "mols": mols})
        dm.to_sdf(pre_processed, f"data/{args.dataset}/preprocessed.sdf", mol_column="mols")
        #save the SMILES in a json file
        with open(f"data/{args.dataset}/smiles.json", "w") as f:
            json.dump(smiles, f)


    else:
        pre_processed = dm.read_sdf(f"data/{args.dataset}/preprocessed.sdf", as_df=True, mol_column="mols")
        smiles = pre_processed["smiles"].iloc[:,0].tolist()
        mols = pre_processed["mols"].tolist()

    for desc in tqdm(DESCRIPTORS, position=0, desc="Descriptors"):
        for length in tqdm(
            [256, 512, 1024, 2048], desc="Length", position=1, leave=False
        ):
            if not os.path.exists(f"data/{args.dataset}/{desc}_{length}.npy"):
                descriptor = get_features(
                    None, smiles, transformer_name =desc, mols=mols, length=length, dataset=args.dataset
                ).numpy()


                np.save(f"data/{args.dataset}/{desc}_{length}.npy", descriptor)
                del descriptor


if __name__ == "__main__":
    main()
