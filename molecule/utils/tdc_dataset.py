import pandas as pd
from tdc.single_pred import Tox, ADME, HTS, QM
from tdc.multi_pred import DTI
from tdc.generation import MolGen

from tdc.utils import retrieve_label_name_list
from tqdm import tqdm

# Correspondancy between dataset name and the corresponding prediction/generation TDC problem
correspondancy_dict = {
    "Tox21": Tox,
    "ToxCast": Tox,
    "LD50_Zhu": Tox,
    "hERG": Tox,
    "herg_central": Tox,
    "hERG_Karim": Tox,
    "AMES": Tox,
    "DILI": Tox,
    "Skin__Reaction": Tox,
    "Skin Reaction": Tox,
    "Carcinogens_Lagunin": Tox,
    "ClinTox": Tox,
    "Caco2_Wang": ADME,
    "PAMPA_NCATS": ADME,
    "HIA_Hou": ADME,
    "Pgp_Broccatelli": ADME,
    "Bioavailability_Ma": ADME,
    "Lipophilicity_AstraZeneca": ADME,
    "Solubility_AqSolDB": ADME,
    "HydrationFreeEnergy_FreeSolv": ADME,
    "BBB_Martins": ADME,
    "PPBR_AZ": ADME,
    "VDss_Lombardo": ADME,
    "CYP2C19_Veith": ADME,
    "CYP2D6_Veith": ADME,
    "CYP3A4_Veith": ADME,
    "CYP1A2_Veith": ADME,
    "CYP2C9_Veith": ADME,
    "CYP2C9_Substrate_CarbonMangels": ADME,
    "CYP2D6_Substrate_CarbonMangels": ADME,
    "CYP3A4_Substrate_CarbonMangels": ADME,
    "Half_Life_Obach": ADME,
    "Clearance_Hepatocyte_AZ": ADME,
    "Clearance_Microsome_AZ": ADME,
    "SARSCoV2_Vitro_Touret": HTS,
    "SARSCoV2_3CLPro_Diamond": HTS,
    "HIV": HTS,
    "orexin1_receptor_butkiewicz": HTS,
    "m1_muscarinic_receptor_agonists_butkiewicz": HTS,
    "m1_muscarinic_receptor_antagonists_butkiewicz": HTS,
    "potassium_ion_channel_kir2.1_butkiewicz": HTS,
    "kcnq2_potassium_channel_butkiewicz": HTS,
    "cav3_t-type_calcium_channels_butkiewicz": HTS,
    "choline_transporter_butkiewicz": HTS,
    "serine_threonine_kinase_33_butkiewicz": HTS,
    "tyrosyl-dna_phosphodiesterase_butkiewicz": HTS,
    "QM7b": QM,
    "QM8": QM,
    "QM9": QM,
    "MOSES": MolGen,
    "ZINC": MolGen,
    "ChEMBL": MolGen,
    "DAVIS": DTI,
    "BindingDB_Kd": DTI,
    "BindingDB_IC50": DTI,
    "KIBA": DTI,
    "BindingDB_Ki": DTI,
}

correspondancy_dict_DTI = {
    "DAVIS": DTI,
    "BindingDB_Kd": DTI,
    "BindingDB_IC50": DTI,
    "KIBA": DTI,
    "BindingDB_Ki": DTI,
}

THRESHOLD_MIN_SAMPLES = 128


def get_dataset(dataset: str):
    try:
        data = correspondancy_dict[dataset](name=dataset)
        if dataset in correspondancy_dict_DTI.keys():
            data.convert_to_log(
                form="binding"
            )
            df = data.harmonize_affinities(mode="max_affinity")
        else:
            df = data.get_data()

    except Exception as e:
        if e.args[0].startswith(
            "Please select a label name. You can use tdc.utils.retrieve_label_name_list"
        ):
            label_list = retrieve_label_name_list(dataset)
            df = []
            for l in tqdm(label_list):
                df.append(
                    correspondancy_dict[dataset](name=dataset, label_name=l).get_data()
                )
            df = pd.concat(df).drop_duplicates("Drug")

    if hasattr(df, "Target_ID"):
        allowed_targets = df["Target_ID"].value_counts()[df["Target_ID"].value_counts() > THRESHOLD_MIN_SAMPLES].index
        df = df[df["Target_ID"].isin(allowed_targets)]
    return df


def get_dataset_split(dataset: str, random_seed: int = 42, method="random"):
    if dataset in correspondancy_dict_DTI.keys():
        return get_dataset_split_DTI(dataset, random_seed=random_seed)
    try:
        split = correspondancy_dict[dataset](name=dataset).get_split(
            seed=random_seed, method=method
        )
        for k in split:
            if split[k].Y.nunique() < 2:
                return get_dataset_split(
                    dataset, random_seed=random_seed + 10, method=method
                )
        return [split]
    except Exception as e:
        if e.args[0].startswith(
            "Please select a label name. You can use tdc.utils.retrieve_label_name_list"
        ):
            label_list = retrieve_label_name_list(dataset)
            split = []
            for l in tqdm(label_list):
                subsplit = correspondancy_dict[dataset](
                    name=dataset, label_name=l
                ).get_split(seed=random_seed, method=method)
                for k in subsplit:
                    if subsplit[k].Y.nunique() < 2:
                        return get_dataset_split(
                            dataset, random_seed=random_seed + 10, method=method
                        )
                split.append(subsplit)
            return split
        else:
            raise e


def get_dataset_split_DTI(dataset, random_seed=42):
    data = correspondancy_dict_DTI[dataset](name=dataset)
    data.convert_to_log(form="binding")
    data = data.harmonize_affinities(mode="max_affinity")
    split = []
    for id in data["Target_ID"].unique():
        subdata = data[data["Target_ID"] == id][["Drug", "Target_ID", "Y"]]
        if subdata.shape[0]<THRESHOLD_MIN_SAMPLES:
            continue
        print(subdata.shape)
        split.append(
            {"train": data[data["Target_ID"] == id][["Drug", "Target_ID", "Y"]]}
        )
    return split
