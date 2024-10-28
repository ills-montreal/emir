moleculenet_models = [
    "Not-trained",
    "AttributeMask",
    "ContextPred",
    "EdgePred",
    "GPT-GNN",
    "InfoGraph",
    "GraphCL",
    "GraphLog",
    "GraphMVP",
    "GROVER",
    "InfoGraph",
]

denoising_models = ["DenoisingPretrainingPQCMv4", "FRAD_QM9"]


class ModelFactory:
    def __new__(cls, name: str):
        if name in moleculenet_models:
            from molecule.utils.embedder_utils.moleculenet import (
                get_embeddings_from_model_moleculenet,
            )

            return get_embeddings_from_model_moleculenet
        elif name in denoising_models:
            from molecule.utils.embedder_utils.denoising_models import (
                get_embeddings_from_model_denoising,
            )

            return get_embeddings_from_model_denoising

        elif name.startswith("MolR"):
            from molecule.utils.embedder_utils.molr import get_embeddings_from_molr

            return get_embeddings_from_molr

        elif name.startswith("MoleOOD"):
            from molecule.utils.embedder_utils.moleood import (
                get_embeddings_from_moleood,
            )

            return get_embeddings_from_moleood

        elif name == "ThreeDInfomax":
            from molecule.utils.embedder_utils.threedinfomax import (
                get_embeddings_from_model_threedinfomax,
            )

            return get_embeddings_from_model_threedinfomax

        else:
            from molecule.utils.embedder_utils.transformers import (
                get_embeddings_from_transformers,
            )

            return get_embeddings_from_transformers
