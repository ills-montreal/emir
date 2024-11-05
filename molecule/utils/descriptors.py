from molecule.utils.embedder_utils.moleculenet_encoding import (
    mol_to_graph_data_obj_simple,
)


DESCRIPTORS = [
    "usrcat",
    "electroshape",
    "usr",
    "ecfp",
    "estate",
    "erg",
    "rdkit",
    "topological",
    "avalon",
    "maccs",
    "atompair-count",
    "topological-count",
    "fcfp-count",
    "secfp",
    "pattern",
    "fcfp",
    "scaffoldkeys",
    "cats",
    "default",
    "gobbi",
    "pmapper",
    "cats/3D",
    "gobbi/3D",
    "pmapper/3D",
]

CONTINUOUS_DESCRIPTORS = [
    "electroshape",
    "usr",
    "usrcat",
]


def can_be_2d_input(smiles, mol):
    if not "." in smiles:
        try:
            _ = mol_to_graph_data_obj_simple(mol)
            return True
        except:
            pass
    return False
