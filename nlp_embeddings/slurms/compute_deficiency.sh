# datasets:

# ./compute_mis_target.sh ../mis_graph/output_norma_3/mis_evaluation ../output ../mis_graph/output_norma_3/logs

output_dir=$1 # ../mis_graph/output_norma_2/mis_evaluation
embeddings_dir=$2 # ../output
log_dir=$3 # ../mis_graph/output_norma_2/logs


# gan_batch_size: int = 64
# gan_n_epochs: int = 100
# critic_repeats: int = 5
#
# # Generator hyperparameters
# gen_hidden_dim: int = 16
# gen_n_layers: int = 5
# gen_lr: float = 0.0001
#
# # Critic hyperparameters
# critic_hidden_dim: int = 16
# critic_n_layers: int = 3
# critic_lr: float = 0.001
#
# # noise hyperparameters
# noise_dim: int = 16
# noise_std = 0.1
#
# # Wasserstein hyperparameters
# disc_clip: float = 0.01
#
# device: str = "cuda:0"


for model_x in  izhx/udever-bloom-560m sentence-transformers/all-MiniLM-L6-v2 jinaai/jina-embedding-s-en-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-mpnet-base-v2 intfloat/e5-small intfloat/multilingual-e5-small sentence-transformers/LaBSE BAAI/bge-base-en-v1.5 sentence-transformers/gtr-t5-xl sentence-transformers/gtr-t5-base avsolatorio/GIST-Embedding-v0 sentence-transformers/gtr-t5-large sentence-transformers/sentence-t5-xl thenlper/gte-base sentence-transformers/sentence-t5-large jamesgpt1/sf_model_e5 infgrad/stella-base-en-v2 thenlper/gte-large intfloat/e5-large-v2 sentence-transformers/average_word_embeddings_komninos sentence-transformers/average_word_embeddings_glove.6B.300d SmartComponents/bge-micro-v2 TaylorAI/gte-tiny sentence-transformers/msmarco-bert-co-condensor princeton-nlp/sup-simcse-bert-base-uncased sentence-transformers/allenai-specter WhereIsAI/UAE-Large-V1 llmrails/ember-v1  "Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B" "jspringer/echo-mistral-7b-instruct-lasttoken" "NousResearch/Llama-2-7b-hf" "togethercomputer/LLaMA-2-7B-32K" "google/gemma-7b" "google/gemma-7b-it" "croissantllm/base_5k" "croissantllm/base_50k" "croissantllm/base_100k" "croissantllm/base_150k" "croissantllm/CroissantCool" "HuggingFaceM4/tiny-random-LlamaForCausalLM" "croissantllm/CroissantLLMBase" "google/gemma-2b" "google/gemma-2b-it"  "google/gemma-7b" "google/gemma-7b-it"; do
for model_Y in  izhx/udever-bloom-560m sentence-transformers/all-MiniLM-L6-v2 jinaai/jina-embedding-s-en-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-mpnet-base-v2 intfloat/e5-small intfloat/multilingual-e5-small sentence-transformers/LaBSE BAAI/bge-base-en-v1.5 sentence-transformers/gtr-t5-xl sentence-transformers/gtr-t5-base avsolatorio/GIST-Embedding-v0 sentence-transformers/gtr-t5-large sentence-transformers/sentence-t5-xl thenlper/gte-base sentence-transformers/sentence-t5-large jamesgpt1/sf_model_e5 infgrad/stella-base-en-v2 thenlper/gte-large intfloat/e5-large-v2 sentence-transformers/average_word_embeddings_komninos sentence-transformers/average_word_embeddings_glove.6B.300d SmartComponents/bge-micro-v2 TaylorAI/gte-tiny sentence-transformers/msmarco-bert-co-condensor princeton-nlp/sup-simcse-bert-base-uncased sentence-transformers/allenai-specter WhereIsAI/UAE-Large-V1 llmrails/ember-v1  "Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B" "jspringer/echo-mistral-7b-instruct-lasttoken" "NousResearch/Llama-2-7b-hf" "togethercomputer/LLaMA-2-7B-32K" "google/gemma-7b" "google/gemma-7b-it" "croissantllm/base_5k" "croissantllm/base_50k" "croissantllm/base_100k" "croissantllm/base_150k" "croissantllm/CroissantCool" "HuggingFaceM4/tiny-random-LlamaForCausalLM" "croissantllm/CroissantLLMBase" "google/gemma-2b" "google/gemma-2b-it"  "google/gemma-7b" "google/gemma-7b-it"; do
  # for dataset in "mteb/amazon_polarity/test/embeddings.npy" "tweet_eval;emoji/test/embeddings.npy" "ag_news/train/embeddings.npy" ; do
  for dataset in "Common"; do
    sbatch --job-name=mis \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=15:00:00 \
      --output=jobinfo_mis/testlib%j.out \
      --error=jobinfo_mis/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        export WANDB_MODE=offline;
        python ../scripts/evaluate_mis_target.py \
        --model_Y ${model_Y} \
        --model_X ${model_x} \
        --output_dir ${output_dir} \
        --embeddings_dir ${embeddings_dir} \
        --log_dir ${log_dir} \

        --gan_batch_size 64 \
        --gan_n_epochs 100 \
        --critic_repeats 5 \

        --gen_hidden_dim 16 \
        --gen_n_layers 5 \
        --gen_lr 0.0001 \

        --critic_hidden_dim 16 \
        --critic_n_layers 3 \
        --critic_lr 0.001 \

        --noise_dim 16 \
        --noise_std 0.1 \

        --disc_clip 0.01 \

        --device cuda \
        --fixed_embeddings ${dataset}
        "
  done
  done
  done

# cmd example with two different model_X

# python ../scripts/evaluate_mis_target.py --model_Y sentence-transformers/all-MiniLM-L6-v2 --model_X  jinaai/jina-embedding-s-en-v1 sentence-transformers/all-distilroberta-v1 --output_dir ../mis_graph/output_norma_2/mis_evaluation --embeddings_dir ../output --log_dir ../mis_graph/output_norma_2/logs --device cuda --batch_size 128  --cond_modes 8 --marg_modes 8 --ff_layers 2 --eval_batch_size 1024 --n_epochs 5 --n_epochs_marg 10 --lr 0.0001 --margin_lr 0.001 --normalize_embeddings --ff_layer_norm --optimize_mu --cov_diagonal var --average var

# "izhx/udever-bloom-560m sentence-transformers/all-MiniLM-L6-v2 jinaai/jina-embedding-s-en-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-mpnet-base-v2 intfloat/e5-small intfloat/multilingual-e5-small sentence-transformers/LaBSE BAAI/bge-base-en-v1.5 sentence-transformers/gtr-t5-xl sentence-transformers/gtr-t5-base avsolatorio/GIST-Embedding-v0 sentence-transformers/gtr-t5-large sentence-transformers/sentence-t5-xl thenlper/gte-base sentence-transformers/sentence-t5-large jamesgpt1/sf_model_e5 infgrad/stella-base-en-v2 thenlper/gte-large intfloat/e5-large-v2 sentence-transformers/average_word_embeddings_komninos sentence-transformers/average_word_embeddings_glove.6B.300d SmartComponents/bge-micro-v2 TaylorAI/gte-tiny sentence-transformers/msmarco-bert-co-condensor princeton-nlp/sup-simcse-bert-base-uncased sentence-transformers/allenai-specter WhereIsAI/UAE-Large-V1 llmrails/ember-v1 Salesforce/SFR-Embedding-Mistral GritLM/GritLM-7B jspringer/echo-mistral-7b-instruct-lasttoken NousResearch/Llama-2-7b-hf togethercomputer/LLaMA-2-7B-32K google/gemma-7b google/gemma-7b-it croissantllm/base_5k croissantllm/base_50k croissantllm/base_100k croissantllm/base_150k croissantllm/CroissantCool HuggingFaceM4/tiny-random-LlamaForCausalLM croissantllm/CroissantLLMBase google/gemma-2b google/gemma-2b-it google/gemma-7b google/gemma-7b-it"
# make a python list of models, add brackets and commas
# "izhx/udever-bloom-560m sentence-transformers/all-MiniLM-L6-v2 jinaai/jina-embedding-s-en-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-mpnet-base-v2 intfloat/e5-small intfloat/multilingual-e5-small sentence-transformers/LaBSE BAAI/bge-base-en-v1.5 sentence-transformers/gtr-t5-xl sentence-transformers/gtr-t5-base avsolatorio/GIST-Embedding-v0 sentence-transformers/gtr-t5-large sentence-transformers/sentence-t5-xl thenlper/gte-base sentence-transformers/sentence-t5-large jamesgpt1/sf_model_e5 infgrad/stella-base-en-v2 thenlper/gte-large intfloat/e5-large-v2 sentence-transformers/average_word_embeddings_komninos sentence-transformers/average_word_embeddings_glove.6B.300d SmartComponents/bge-micro-v2 TaylorAI/gte-tiny sentence-transformers/msmarco-bert-co-condensor princeton-nlp/sup-simcse-bert-base-uncased sentence-transformers/allenai-specter WhereIsAI/UAE-Large-V1 llmrails/ember-v1 Salesforce/SFR-Embedding-Mistral GritLM/GritLM-7B jspringer/echo-mistral-7b-instruct-lasttoken NousResearch/Llama-2-7b-hf togethercomputer/LLaMA-2-7B-32K google/gemma-7b google/gemma-7b-it croissantllm/base_5k croissantllm/base_50k croissantllm/base_100k croissantllm/base_150k croissantllm/CroissantCool HuggingFaceM4/tiny-random-LlamaForCausalLM croissantllm/CroissantLLMBase google/gemma-2b google/gemma-2b-it google/gemma-7b google/gemma-7b-it"
# make a python list of models, add brackets and commas
