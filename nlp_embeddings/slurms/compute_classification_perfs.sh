# datasets:

output_dir=$1
embeddings_dir=$2

for model_1 in  'izhx/udever-bloom-560m' 'sentence-transformers/all-MiniLM-L6-v2' 'jinaai/jina-embedding-s-en-v1' 'sentence-transformers/all-distilroberta-v1' 'sentence-transformers/all-mpnet-base-v2' 'intfloat/e5-small' 'intfloat/multilingual-e5-small' 'sentence-transformers/LaBSE' 'BAAI/bge-base-en-v1.5' 'sentence-transformers/gtr-t5-xl' 'sentence-transformers/gtr-t5-base' 'avsolatorio/GIST-Embedding-v0' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/sentence-t5-xl' 'thenlper/gte-base' 'sentence-transformers/sentence-t5-large' 'jamesgpt1/sf_model_e5' 'infgrad/stella-base-en-v2' 'thenlper/gte-large' 'intfloat/e5-large-v2' 'sentence-transformers/average_word_embeddings_komninos' 'sentence-transformers/average_word_embeddings_glove.6B.300d' 'SmartComponents/bge-micro-v2' 'TaylorAI/gte-tiny' 'sentence-transformers/msmarco-bert-co-condensor' 'princeton-nlp/sup-simcse-bert-base-uncased' 'sentence-transformers/allenai-specter' 'WhereIsAI/UAE-Large-V1' 'llmrails/ember-v1'; do
<<<<<<< HEAD
for dataset in "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "clinc_oos;plus" "dair-ai/emotion"  "sst2" "rotten_tomatoes" "imdb" "ag_news" "dair-ai/emotion" "paws-x;en"; do
=======
for dataset in rotten_tomatoes sst2; do
>>>>>>> 33a03ab (fix kinfe estimator)
    sbatch --job-name=classif \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=00:15:00 \
      --output=jobinfo_classif/testlib%j.out \
      --error=jobinfo_classif/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        export WANDB_MODE=offline;
        python ../scripts/train_eval_embedding_for_classification.py \
        --model ${model_1} \
<<<<<<< HEAD
        --dataset '${dataset}' \
=======
        --dataset ${dataset} \
>>>>>>> 33a03ab (fix kinfe estimator)
        --output_dir ${output_dir} \
        --embeddings_dir ${embeddings_dir} \
        --n_epochs 2
        "
  done
done
