# datasets:

output_dir=$1
embeddings_dir=$2
log_dir=$3

for model_1 in  'izhx/udever-bloom-560m' 'sentence-transformers/all-MiniLM-L6-v2' 'jinaai/jina-embedding-s-en-v1' 'sentence-transformers/all-distilroberta-v1' 'sentence-transformers/all-mpnet-base-v2' 'intfloat/e5-small' 'intfloat/multilingual-e5-small' 'sentence-transformers/LaBSE' 'BAAI/bge-base-en-v1.5' 'sentence-transformers/gtr-t5-xl' 'sentence-transformers/gtr-t5-base' 'avsolatorio/GIST-Embedding-v0' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/sentence-t5-xl' 'thenlper/gte-base' 'sentence-transformers/sentence-t5-large' 'jamesgpt1/sf_model_e5' 'infgrad/stella-base-en-v2' 'thenlper/gte-large' 'intfloat/e5-large-v2' 'sentence-transformers/average_word_embeddings_komninos' 'sentence-transformers/average_word_embeddings_glove.6B.300d' 'SmartComponents/bge-micro-v2' 'TaylorAI/gte-tiny' 'sentence-transformers/msmarco-bert-co-condensor' 'princeton-nlp/sup-simcse-bert-base-uncased' 'sentence-transformers/allenai-specter' 'WhereIsAI/UAE-Large-V1' 'llmrails/ember-v1'; do
for model_2 in  'izhx/udever-bloom-560m' 'sentence-transformers/all-MiniLM-L6-v2' 'jinaai/jina-embedding-s-en-v1' 'sentence-transformers/all-distilroberta-v1' 'sentence-transformers/all-mpnet-base-v2' 'intfloat/e5-small' 'intfloat/multilingual-e5-small' 'sentence-transformers/LaBSE' 'BAAI/bge-base-en-v1.5' 'sentence-transformers/gtr-t5-xl' 'sentence-transformers/gtr-t5-base' 'avsolatorio/GIST-Embedding-v0' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/sentence-t5-xl' 'thenlper/gte-base' 'sentence-transformers/sentence-t5-large' 'jamesgpt1/sf_model_e5' 'infgrad/stella-base-en-v2' 'thenlper/gte-large' 'intfloat/e5-large-v2' 'sentence-transformers/average_word_embeddings_komninos' 'sentence-transformers/average_word_embeddings_glove.6B.300d' 'SmartComponents/bge-micro-v2' 'TaylorAI/gte-tiny' 'sentence-transformers/msmarco-bert-co-condensor' 'princeton-nlp/sup-simcse-bert-base-uncased' 'sentence-transformers/allenai-specter' 'WhereIsAI/UAE-Large-V1' 'llmrails/ember-v1'; do
    sbatch --job-name=crossemb \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=1:00:00 \
      --output=jobinfo_crossemb/testlib%j.out \
      --error=jobinfo_crossemb/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1; export WANDB_MODE=offline;
        python ../scripts/train_eval_cross_embeddings_prediction.py \
        --model_1 ${model_1} \
        --model_2 ${model_2} \
        --output_dir ${output_dir} \
        --embeddings_dir ${embeddings_dir} \
        --n_epochs 2
        "
  done
done
