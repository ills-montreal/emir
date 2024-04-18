# datasets:

output_dir=$1
embeddings_dir=$2

for model in 'llmrails/ember-v1 infgrad/stella-base-en-v2' 'infgrad/stella-base-en-v2 intfloat/e5-large-v2' 'infgrad/stella-base-en-v2 sentence-transformers/gtr-t5-large' 'llmrails/ember-v1 intfloat/e5-large-v2' 'infgrad/stella-base-en-v2 TaylorAI/gte-tiny' 'llmrails/ember-v1 TaylorAI/gte-tiny' 'llmrails/ember-v1 sentence-transformers/gtr-t5-large' 'intfloat/e5-large-v2 sentence-transformers/gtr-t5-large' 'intfloat/e5-large-v2 TaylorAI/gte-tiny' 'sentence-transformers/gtr-t5-large TaylorAI/gte-tiny'; do
for dataset in "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "clinc_oos;plus" "dair-ai/emotion"  "sst2" "rotten_tomatoes" "imdb" "ag_news" "dair-ai/emotion" "paws-x;en"; do
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
        --model ${model} \
        --dataset '${dataset}' \
        --output_dir ${output_dir} \
        --embeddings_dir ${embeddings_dir} \
        --n_epochs 2
        "
  done
done

# example
# python scripts/train_eval_embedding_for_classification.py --model llmrails/ember-v1 infgrad/stella-base-en-v2 --dataset 'tweet_eval;emoji' --output_dir ../test/concat --embeddings_dir classification_embeddings --n_epochs 2