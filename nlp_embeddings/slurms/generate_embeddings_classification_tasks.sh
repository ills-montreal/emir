# datasets:

output_dir=$1

#TASKS_DATASET = {
#    "yelp_review_full": {
#        "dataset_name": "yelp_review_full",
#        "config": None,
#        "task": "Sentiment classification",
#        "num_classes": 5,
#    },  # text, label
#    "paws-x;en": {
#        "dataset_name": "paws-x",
#        "config": "en",
#        "task": "Paraphrase identification",
#        "num_classes": 2,
#    },  # sentence1, sentence2, label
#    "sst2": {
#        "dataset_name": "sst2",
#        "config": None,
#        "task": "Sentiment classification",
#        "num_classes": 2,
#    },  # sentence, label
#    "tweet_eval;emoji": {
#        "dataset_name": "tweet_eval",
#        "config": "emoji",
#        "task": "Emoji prediction",
#        "n_classes": 20,
#    },  # text, label
#    "tweet_eval;emotion": {
#        "dataset_name": "tweet_eval",
#        "config": "emotion",
#        "task": "Emotion prediction",
#        "n_classes": 4,
#    },  # text, label
#    "tweet_eval;sentiment": {
#        "dataset_name": "tweet_eval",
#        "config": "sentiment",
#        "task": "Sentiment prediction",
#        "n_classes": 3,
#    },  # text, label
#    "rotten_tomatoes": {
#        "dataset_name": "rotten_tomatoes",
#        "config": None,
#        "task": "Sentiment classification",
#        "num_classes": 2,
#    },  # text, label
#    "imdb": {
#        "dataset_name": "imdb",
#        "config": None,
#        "task": "Sentiment classification",
#        "num_classes": 2,
#    },  # text, label
#    "clinc_oos;plus": {
#        "dataset_name": "clinc_oos",
#        "config": "plus",
#        "task": "Intent classification",
#        "num_classes": 151,
#    },  # text, intent
#}
<<<<<<< HEAD
# "yelp_review_full" "paws-x;en" "sst2" "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "rotten_tomatoes" "imdb" "clinc_oos;plus" "banking77" "ag_news" "dair-ai/emotion/"; do
# "clinc_oos;plus" "dair-ai/emotion"

# "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "clinc_oos;plus" "dair-ai/emotion" "paws-x;en" "sst2" "rotten_tomatoes" "imdb" "ag_news" "dair-ai/emotion"
# "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment"; do
for model in BAAI/bge-base-en-v1.5 infgrad/stella-base-en-v2 intfloat/e5-large-v2 intfloat/multilingual-e5-small sentence-transformers/sentence-t5-xl sentence-transformers/sentence-t5-large SmartComponents/bge-micro-v2 sentence-transformers/allenai-specter sentence-transformers/average_word_embeddings_glove.6B.300d sentence-transformers/average_word_embeddings_komninos sentence-transformers/LaBSE avsolatorio/GIST-Embedding-v0 Muennighoff/SGPT-125M-weightedmean-nli-bitfit princeton-nlp/sup-simcse-bert-base-uncased jinaai/jina-embedding-s-en-v1 sentence-transformers/msmarco-bert-co-condensor sentence-transformers/gtr-t5-base izhx/udever-bloom-560m llmrails/ember-v1 jamesgpt1/sf_model_e5 thenlper/gte-large TaylorAI/gte-tiny sentence-transformers/gtr-t5-xl intfloat/e5-small sentence-transformers/gtr-t5-large thenlper/gte-base nomic-ai/nomic-embed-text-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2; do
  for dataset in "paws-x;en"; do
=======

for model in BAAI/bge-base-en-v1.5 infgrad/stella-base-en-v2 intfloat/e5-large-v2 intfloat/multilingual-e5-small sentence-transformers/sentence-t5-xl sentence-transformers/sentence-t5-large SmartComponents/bge-micro-v2 sentence-transformers/allenai-specter sentence-transformers/average_word_embeddings_glove.6B.300d sentence-transformers/average_word_embeddings_komninos sentence-transformers/LaBSE avsolatorio/GIST-Embedding-v0 Muennighoff/SGPT-125M-weightedmean-nli-bitfit princeton-nlp/sup-simcse-bert-base-uncased jinaai/jina-embedding-s-en-v1 sentence-transformers/msmarco-bert-co-condensor sentence-transformers/gtr-t5-base izhx/udever-bloom-560m llmrails/ember-v1 jamesgpt1/sf_model_e5 thenlper/gte-large TaylorAI/gte-tiny sentence-transformers/gtr-t5-xl intfloat/e5-small sentence-transformers/gtr-t5-large thenlper/gte-base nomic-ai/nomic-embed-text-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2; do
  for dataset in "yelp_review_full" "paws-x;en" "sst2" "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "rotten_tomatoes" "imdb" "clinc_oos;plus"; do
>>>>>>> 33a03ab (fix kinfe estimator)
    sbatch --job-name=emb_classif \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=6:00:00 \
      --output=jobinfo_emb/testlib%j.out \
      --error=jobinfo_emb/testlib%j.err \
<<<<<<< HEAD
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
=======
      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
>>>>>>> 33a03ab (fix kinfe estimator)
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python ../scripts/generate_embeddings.py \
        --model ${model} \
<<<<<<< HEAD
        --dataset '${dataset}' \
=======
        --dataset ${dataset} \
        --classification_task \
        --output_dir ${output_dir} \
        --device cuda \
        --batch_size 256"
  done
done

for model in "WhereIsAI/UAE-Large-V1" "Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B" "jspringer/echo-mistral-7b-instruct-lasttoken"; do
  for dataset in "yelp_review_full" "paws-x;en" "sst2" "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "rotten_tomatoes" "imdb" "clinc_oos;plus"; do
    sbatch --job-name=emb_classif \
      --account=ehz@a100 \
      --gres=gpu:1 \
      --partition=gpu_p5 \
      --no-requeue \
      --cpus-per-task=8 \
      --hint=nomultithread \
      --time=5:00:00 \
      -C a100 \
      --output=jobinfo_emb/testlib%j.out \
      --error=jobinfo_emb/testlib%j.err \
      --wrap="module purge; module load cpuarch/amd ; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python ../scripts/generate_embeddings.py \
        --model ${model} \
        --dataset ${dataset} \
>>>>>>> 33a03ab (fix kinfe estimator)
        --classification_task \
        --output_dir ${output_dir} \
        --device cuda \
        --batch_size 256"
  done
done
<<<<<<< HEAD
# "yelp_review_full" "paws-x;en" "sst2" "tweet_eval;emoji" "tweet_eval;emotion" "tweet_eval;sentiment" "rotten_tomatoes" "imdb" "clinc_oos;plus"; do
#for model in "WhereIsAI/UAE-Large-V1" "Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B" "jspringer/echo-mistral-7b-instruct-lasttoken"; do
#  for dataset in "clinc_oos;plus" "banking77" "ag_news" "dair-ai/emotion"; do
#    sbatch --job-name=emb_classif \
#      --account=ehz@a100 \
#      --gres=gpu:1 \
#      --partition=gpu_p5 \
#      --no-requeue \
#      --cpus-per-task=8 \
#      --hint=nomultithread \
#      --time=5:00:00 \
#      -C a100 \
#      --output=jobinfo_emb/testlib%j.out \
#      --error=jobinfo_emb/testlib%j.err \
#      --wrap="module purge; module load cpuarch/amd ; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python ../scripts/generate_embeddings.py \
#        --model ${model} \
#        --dataset '${dataset}' \
#        --classification_task \
#        --output_dir ${output_dir} \
#        --device cuda \
#        --batch_size 256"
#  done
#done
=======
>>>>>>> 33a03ab (fix kinfe estimator)
