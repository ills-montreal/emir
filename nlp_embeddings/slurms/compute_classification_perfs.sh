# datasets:

output_dir=$1
embeddings_dir=$2

for model_1 in  "BAAI/bge-base-en-v1.5" "GritLM/GritLM-7B" "HuggingFaceM4/tiny-random-LlamaForCausalLM" "NousResearch/Llama-2-7b-hf" "Salesforce/SFR-Embedding-Mistral" "SmartComponents/bge-micro-v2" "TaylorAI/gte-tiny" "WhereIsAI/UAE-Large-V1" "avsolatorio/GIST-Embedding-v0" "croissantllm/CroissantCool" "croissantllm/CroissantLLMBase" "croissantllm/base_100k" "croissantllm/base_150k" "croissantllm/base_50k" "croissantllm/base_5k" "google/gemma-2b" "google/gemma-2b-it" "google/gemma-7b" "google/gemma-7b-it" "infgrad/stella-base-en-v2" "intfloat/e5-large-v2" "intfloat/e5-small" "intfloat/multilingual-e5-small" "izhx/udever-bloom-560m" "jamesgpt1/sf_model_e5" "jspringer/echo-mistral-7b-instruct-lasttoken" "llmrails/ember-v1" "princeton-nlp/sup-simcse-bert-base-uncased" "sentence-transformers/LaBSE" "sentence-transformers/all-MiniLM-L6-v2" "sentence-transformers/all-distilroberta-v1" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/allenai-specter" "sentence-transformers/average_word_embeddings_glove.6B.300d" "sentence-transformers/average_word_embeddings_komninos" "sentence-transformers/gtr-t5-base" "sentence-transformers/gtr-t5-large" "sentence-transformers/gtr-t5-xl" "sentence-transformers/msmarco-bert-co-condensor" "sentence-transformers/sentence-t5-large" "sentence-transformers/sentence-t5-xl" "thenlper/gte-base" "thenlper/gte-large"; do
for dataset in "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "clinc_oos;plus" "dair-ai/emotion"  "sst2" "rotten_tomatoes" "imdb" "ag_news" "dair-ai/emotion" "paws-x;en"; do
    sbatch --job-name=classif \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=01:00:00 \
      --output=jobinfo_classif/testlib%j.out \
      --error=jobinfo_classif/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        export WANDB_MODE=offline;
        python ../scripts/train_eval_embedding_for_classification.py \
        --model ${model_1} \
        --dataset '${dataset}' \
        --output_dir ${output_dir} \
        --embeddings_dir ${embeddings_dir} \
        --n_epochs 2
        "
  done
done
