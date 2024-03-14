search_dir=$1
output_dir=$2
for model in 1 2 3 4 5 6; do
  for entry in "$search_dir"/*; do
    sbatch --job-name=shm \
      --account=ehz@a100 \
      --gres=gpu:1 \
      --partition=gpu_p5 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=5:00:00 \
      -C a100 \
      --output=jobinfo_eval_shm/testlib%j.out \
      --error=jobinfo_eval_shm/testlib%j.err \
      --wrap="module purge; module load cpuarch/amd ; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/evaluate_seahorse_metrics.py \
        --question ${model} \
        --batch_size 2 \
        --summaries ${entry} \
        --output ${output_dir} \
        --device cuda"
  done
done
