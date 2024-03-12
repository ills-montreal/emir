search_dir=$1
output_dir=$2
# for model in #"manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1" "jonaskoenig/topic_classification_04"; do
for model in sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2; do
# Falconsai/intent_classification; do
# # wesleyacheng/news-topic-classification-with-bert cardiffnlp/tweet-topic-21-multi SamLowe/roberta-base-go_emotions mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis roberta-base-openai-detector roberta-large-openai-detector; do
  for entry in "$search_dir"/*; do
    sbatch --job-name=emb \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=6:00:00 \
      --output=jobinfo_eval_emb/testlib%j.out \
      --error=jobinfo_eval_emb/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/evaluate_embeddings.py \
        --model ${model} \
        --summaries ${entry} \
        --output ${output_dir} \
        --device cuda"
  done
done
