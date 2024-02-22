search_dir=$1
for model in wesleyacheng/news-topic-classification-with-bert cardiffnlp/tweet-topic-21-multi SamLowe/roberta-base-go_emotions mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis roberta-base-openai-detector roberta-large-openai-detector; do
  for entry in "$search_dir"/*; do
    sbatch --job-name=cs \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=6:00:00 \
      --output=jobinfo_eval_classif_samples/testlib%j.out \
      --error=jobinfo_eval_classif_samples/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/evaluate_classification_task_samples.py \
        --model ${model} \
        --summaries ${entry} \
        --device cuda"
  done
done
