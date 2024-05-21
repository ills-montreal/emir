from sentence_transformers import SentenceTransformer

SMALL_MODELS = [
    "Snowflake/snowflake-arctic-embed-s",
    "Snowflake/snowflake-arctic-embed-xs",
    "Snowflake/snowflake-arctic-embed-m",
    "Snowflake/snowflake-arctic-embed-l",
    # "BAAI/bge-base-en-v1.5",
    # "infgrad/stella-base-en-v2",
    # "intfloat/e5-large-v2",
    # "intfloat/multilingual-e5-small",
    # "sentence-transformers/sentence-t5-xl",
    # "sentence-transformers/sentence-t5-large",
    # "SmartComponents/bge-micro-v2",
    # "sentence-transformers/allenai-specter",
    # "sentence-transformers/average_word_embeddings_glove.6B.300d",
    # "sentence-transformers/average_word_embeddings_komninos",
    # "sentence-transformers/LaBSE",
    # "avsolatorio/GIST-Embedding-v0",
    # "Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    # "princeton-nlp/sup-simcse-bert-base-uncased",
    # "jinaai/jina-embedding-s-en-v1",
    # "sentence-transformers/msmarco-bert-co-condensor",
    # "sentence-transformers/gtr-t5-base",
    # "izhx/udever-bloom-560m",
    # "llmrails/ember-v1",
    # "jamesgpt1/sf_model_e5",
    # "thenlper/gte-large",
    # "TaylorAI/gte-tiny",
    # "sentence-transformers/gtr-t5-xl",
    # "intfloat/e5-small",
    # "sentence-transformers/gtr-t5-large",
    # "thenlper/gte-base",
    # "sentence-transformers/all-distilroberta-v1",
    # "sentence-transformers/all-MiniLM-L6-v2",
    # "sentence-transformers/all-mpnet-base-v2",
]

LARGE_MODELS = [
    "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    # "croissantllm/base_5k",
    # "croissantllm/base_50k",
    # "croissantllm/base_100k",
    # "croissantllm/base_150k",
    # "croissantllm/CroissantCool",
    # "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    # "croissantllm/CroissantLLMBase",
    # "NousResearch/Llama-2-7b-hf",
    # "togethercomputer/LLaMA-2-7B-32K",
    # "google/gemma-7b",
    # "google/gemma-2b",
    # "google/gemma-7b-it",
    # "google/gemma-2b-it",
    # "WhereIsAI/UAE-Large-V1",
    # "Salesforce/SFR-Embedding-Mistral",
    # "GritLM/GritLM-7B",
    # "jspringer/echo-mistral-7b-instruct-lasttoken",
]


if __name__ == "__main__":
    # for model in SMALL_MODELS:
    #     print(f"Downloading {model}")
    #     SentenceTransformer(model, trust_remote_code=True, device="cpu")
    for model in SMALL_MODELS + LARGE_MODELS:
        print(f"Downloading {model}")
        SentenceTransformer(model, trust_remote_code=True, device="cpu")
