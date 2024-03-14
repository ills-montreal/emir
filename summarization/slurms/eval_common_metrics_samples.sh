search_dir=$1
for entry in "$search_dir"/*; do
  sbatch --job-name=common \
    --account=ehz@cpu \
    --no-requeue \
    --cpus-per-task=2 \
    --hint=nomultithread \
    --time=5:00:00 \
    --output=jobinfo_eval_common_samples/testlib%j.out \
    --error=jobinfo_eval_common_samples/testlib%j.err \
    --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/evaluate_common_metrics_samples.py --summaries ${entry}"
done

