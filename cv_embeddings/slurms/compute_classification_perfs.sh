# datasets:

output_dir=$1
embeddings_dir=$2

for model_1 in  "BAAI/bge-base-en-v1.5" "GritLM/GritLM-7B" "HuggingFaceM4/tiny-random-LlamaForCausalLM"; do
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
