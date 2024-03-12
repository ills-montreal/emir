from sentence_transformers import SentenceTransformer

SMALL_MODELS = [
    "avsolatorio/GIST-Embedding-v0",
    "llmrails/ember-v1",
    "jamesgpt1/sf_model_e5",
    "thenlper/gte-large",
    "avsolatorio/GIST-small-Embedding-v0",
    "thenlper/gte-base",
    "nomic-ai/nomic-embed-text-v1",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

LARGE_MODELS = [
    "WhereIsAI/UAE-Large-V1",
    "Salesforce/SFR-Embedding-Mistral",
    "GritLM/GritLM-7B",
    "jspringer/echo-mistral-7b-instruct-lasttoken",
]


if __name__ == "__main__":
    for model in SMALL_MODELS:
        print(f"Downloading {model}")
        SentenceTransformer(model, trust_remote_code=True, device="cpu")
    for model in LARGE_MODELS:
        print(f"Downloading {model}")
        SentenceTransformer(model, trust_remote_code=True, device="cpu")
