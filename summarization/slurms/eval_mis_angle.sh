search_dir=$1
output_dir=$2
# --model paraphrase-MiniLM-L6-v2 \
for entry in "$search_dir"/*; do
      sbatch --job-name=qa \
        --account=ehz@v100 \
        --gres=gpu:1 \
        --partition=gpu_p2 \
        --no-requeue \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=5:00:00 \
        --output=jobinfo_eval/testlib%j.out \
        --error=jobinfo_eval/testlib%j.err \
        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/evaluate_summarizer_angle.py \
        --summaries ${entry} \
        --output  ${output_dir} \
        --model WhereIsAI/UAE-Large-V1 \
        --batch_size 32 \
        --n_epochs 1000 \
        --stopping_criterion early_stopping \
        --eps 1e-6 \
        --lr 0.001 \
        --device cuda \
        --cond_modes 4 \
        --marg_modes 4"
done
