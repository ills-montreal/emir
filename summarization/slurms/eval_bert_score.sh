search_dir=$1
output_dir=$2
for entry in "$search_dir"/*; do
  sbatch --job-name=bert \
    --account=ehz@v100 \
    --gres=gpu:1 \
    --no-requeue \
    --cpus-per-task=2 \
    --hint=nomultithread \
    --time=5:00:00 \
    --output=jobinfo_eval_bert/testlib%j.out \
    --error=jobinfo_eval_bert/testlib%j.err \
    --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/evaluate_bartbert_metrics.py \
        --summaries ${entry} \
        --output ${output_dir}"
done

