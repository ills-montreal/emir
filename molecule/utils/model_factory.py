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


class ModelFactory:
    def __new__(cls, name: str):
        if name in moleculenet_models:
            from .moleculenet import get_embeddings_from_model_moleculenet

            return get_embeddings_from_model_moleculenet
        else:
            from .transformers import get_embeddings_from_transformers

            return get_embeddings_from_transformers
