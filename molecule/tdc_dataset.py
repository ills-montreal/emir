from tdc.single_pred import Tox, ADME, HTS, QM
from tdc.generation import MolGen

from tdc.utils import retrieve_label_name_list

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
}


def get_dataset(dataset: str):
    try:
        df = correspondancy_dict[dataset](name=dataset).get_data()
    except Exception as e:
        label_list = retrieve_label_name_list(dataset)
        df = correspondancy_dict[dataset](name=dataset, label_name=label_list[0]).get_data()
    return df