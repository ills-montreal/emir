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
            from .moleculenet import get_embeddings_from_model_moleculenet

            return get_embeddings_from_model_moleculenet
        elif name in denoising_models:
            from .denoising_models import get_embeddings_from_model_denoising

            return get_embeddings_from_model_denoising

        elif name.startswith("MolR"):
            from .molr import get_embeddings_from_molr

            return get_embeddings_from_molr

        else:
            from .transformers import get_embeddings_from_transformers

            return get_embeddings_from_transformers
