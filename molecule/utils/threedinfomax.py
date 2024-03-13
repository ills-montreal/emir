from molecule.models.threedinfomax import ThreeDInfomax


def get_embeddings_from_model_threedinfomax(
    smiles: List[str],
    transformer_name: str = "",
    batch_size: int = 2048,
    device: str = "cpu",
    **kwargs,
):
    model = ThreeDInfomax(smiles, device=device, batch_size=batch_size)
    embeddings = model.featurize_data(smiles)
    return embeddings
