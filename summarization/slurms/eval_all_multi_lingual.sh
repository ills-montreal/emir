
search_dir="output/summaries/multilingual_large/"
output_dir="output/metric_evaluation/multilingual_large"

## for model in #"manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1" "jonaskoenig/topic_classification_04"; do
#for model in laiyer/deberta-v3-base-prompt-injection ProsusAI/finbert wesleyacheng/news-topic-classification-with-bert cardiffnlp/tweet-topic-21-multi SamLowe/roberta-base-go_emotions mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis roberta-base-openai-detector roberta-large-openai-detector "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1" "jonaskoenig/topic_classification_04"; do
## Falconsai/intent_classification; do
## # wesleyacheng/news-topic-classification-with-bert cardiffnlp/tweet-topic-21-multi SamLowe/roberta-base-go_emotions mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis roberta-base-openai-detector roberta-large-openai-detector; do
#  for entry in "$search_dir"/*; do
#    sbatch --job-name=qa \
#      --account=ehz@v100 \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --no-requeue \
#      --cpus-per-task=10 \
#      --hint=nomultithread \
#      --time=6:00:00 \
#      --output=jobinfo_eval_classif/testlib%j.out \
#      --error=jobinfo_eval_classif/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python EMIR/summarization/summarization_evaluation/evaluate_classification_task.py \
#        --model ${model} \
#        --summaries ${entry} \
#        --output ${output_dir}_classifiers \
#        --device cuda"
#  done
#done
#
#
#for entry in "$search_dir"/*; do
#  sbatch --job-name=sh_questions \
#    --account=ehz@v100 \
#    --gres=gpu:1 \
#    --no-requeue \
#    --cpus-per-task=2 \
#    --hint=nomultithread \
#    --time=5:00:00 \
#    --output=jobinfo_eval_bert/testlib%j.out \
#    --error=jobinfo_eval_bert/testlib%j.err \
#    --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python EMIR/summarization/summarization_evaluation/evaluate_bartbert_metrics.py \
#        --summaries ${entry} \
#        --output ${output_dir}_bert"
#done
#
#for entry in "$search_dir"/*; do
#  sbatch --job-name=sh_questions \
#    --account=ehz@cpu \
#    --no-requeue \
#    --cpus-per-task=2 \
#    --hint=nomultithread \
#    --time=5:00:00 \
#    --output=jobinfo_eval_sh/testlib%j.out \
#    --error=jobinfo_eval_sh/testlib%j.err \
#    --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python EMIR/summarization/summarization_evaluation/evaluate_common_metrics.py \
#        --summaries ${entry} \
#        --output ${output_dir}_common"
#done
#
#for model in sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2 sentence-transformers/paraphrase-MiniLM-L6-v2; do
#  for entry in "$search_dir"/*; do
#    sbatch --job-name=emb \
#      --account=ehz@v100 \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --no-requeue \
#      --cpus-per-task=10 \
#      --hint=nomultithread \
#      --time=6:00:00 \
#      --output=jobinfo_eval_emb/testlib%j.out \
#      --error=jobinfo_eval_emb/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python EMIR/summarization/summarization_evaluation/evaluate_embeddings.py \
#        --model ${model} \
#        --summaries ${entry} \
#        --output ${output_dir}_emb \
#        --device cuda"
#  done
#done

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
        python EMIR/summarization/summarization_evaluation/evaluate_summarizer.py \
        --summaries ${entry} \
        --output  ${output_dir}_mis3 \
        --model intfloat/multilingual-e5-base \
        --batch_size 32 \
        --n_epochs 1000 \
        --stopping_criterion early_stopping \
        --eps 1e-6 \
        --lr 0.005 \
        --device cuda \
        --cond_modes 4 \
        --marg_modes 4"
done
#
#for model in 1 2 3 4 5 6; do
#  for entry in "$search_dir"/*; do
#    sbatch --job-name=shm \
#      --account=ehz@a100 \
#      --gres=gpu:1 \
#      --partition=gpu_p5 \
#      --no-requeue \
#      --cpus-per-task=10 \
#      --hint=nomultithread \
#      --time=5:00:00 \
#      -C a100 \
#      --output=jobinfo_eval_shm/testlib%j.out \
#      --error=jobinfo_eval_shm/testlib%j.err \
#      --wrap="module purge; module load cpuarch/amd ; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python EMIR/summarization/summarization_evaluation/evaluate_seahorse_metrics.py \
#        --question ${model} \
#        --batch_size 2 \
#        --summaries ${entry} \
#        --output ${output_dir}_shm \
#        --device cuda"
#  done
#done
