# datasets:

#    "mteb/sts12-sts",
#    "mteb/sts13-sts",
#    "mteb/sts14-sts",
#    "mteb/sts15-sts",
#    "meteb/amazon_polarity",
#    "dennlinger/wiki-paragraphs",
#    "mteb/banking77",
#    "mteb/sickr-sts",
#    "mteb/biosses-sts",
#    "mteb/stsbenchmark-sts",
#    "mteb/imdb",
#    "nvidia/OpenMathInstruct-1",
#    "snli",
#    "Open-Orca/OpenOrca",
#    "cnn_dailymail",
#    "EdinburghNLP/xsum",

output_dir=$1
<<<<<<< HEAD
#    "BAAI/bge-base-en-v1.5",
#    "infgrad/stella-base-en-v2",
#    "intfloat/e5-large-v2",
#    "intfloat/multilingual-e5-small",
#    "sentence-transformers/sentence-t5-xl",
#    "sentence-transformers/sentence-t5-large",
#    "SmartComponents/bge-micro-v2",
#    "sentence-transformers/allenai-specter",
#    "sentence-transformers/average_word_embeddings_glove.6B.300d",
#    "sentence-transformers/average_word_embeddings_komninos",
#    "sentence-transformers/LaBSE",
#    "avsolatorio/GIST-Embedding-v0",
#    "Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
#    "princeton-nlp/sup-simcse-bert-base-uncased",
#    "jinaai/jina-embedding-s-en-v1",
#    "sentence-transformers/msmarco-bert-co-condensor",
#    "sentence-transformers/gtr-t5-base",
#    "izhx/udever-bloom-560m",
#    "llmrails/ember-v1",
#    "jamesgpt1/sf_model_e5",
#    "thenlper/gte-large",
#    "TaylorAI/gte-tiny",
#    "sentence-transformers/gtr-t5-xl",
#    "intfloat/e5-small",
#    "sentence-transformers/gtr-t5-large",
#    "thenlper/gte-base",
#    "nomic-ai/nomic-embed-text-v1",
#    "sentence-transformers/all-distilroberta-v1",
#    "sentence-transformers/all-MiniLM-L6-v2",
#    "sentence-transformers/all-mpnet-base-v2",
#]

#for model in BAAI/bge-base-en-v1.5 infgrad/stella-base-en-v2 intfloat/e5-large-v2 intfloat/multilingual-e5-small sentence-transformers/sentence-t5-xl sentence-transformers/sentence-t5-large SmartComponents/bge-micro-v2 sentence-transformers/allenai-specter sentence-transformers/average_word_embeddings_glove.6B.300d sentence-transformers/average_word_embeddings_komninos sentence-transformers/LaBSE avsolatorio/GIST-Embedding-v0 Muennighoff/SGPT-125M-weightedmean-nli-bitfit princeton-nlp/sup-simcse-bert-base-uncased jinaai/jina-embedding-s-en-v1 sentence-transformers/msmarco-bert-co-condensor sentence-transformers/gtr-t5-base izhx/udever-bloom-560m llmrails/ember-v1 jamesgpt1/sf_model_e5 thenlper/gte-large TaylorAI/gte-tiny sentence-transformers/gtr-t5-xl intfloat/e5-small sentence-transformers/gtr-t5-large thenlper/gte-base nomic-ai/nomic-embed-text-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2; do
#  for dataset in "mteb/sts12-sts" "mteb/sts13-sts" "mteb/sts14-sts" "mteb/sts15-sts" "mteb/amazon_polarity" "dennlinger/wiki-paragraphs" "mteb/banking77" "mteb/sickr-sts" "mteb/biosses-sts" "mteb/stsbenchmark-sts" "mteb/imdb" "snli"; do
=======
# sentence-transformers/all-distilroberta-v1  sentence-transformers/all-MiniLM-L6-v2 WhereIsAI/UAE-Large-V1 sentence-transformers/all-mpnet-base-v2

#    "avsolatorio/GIST-Embedding-v0",
#    "llmrails/ember-v1",
#    "jamesgpt1/sf_model_e5",
#    "thenlper/gte-large",
#    "avsolatorio/GIST-small-Embedding-v0",
#    "thenlper/gte-base",
#    "nomic-ai/nomic-embed-text-v1",

#for model in "avsolatorio/GIST-Embedding-v0" "llmrails/ember-v1" "jamesgpt1/sf_model_e5" "thenlper/gte-large" "avsolatorio/GIST-small-Embedding-v0" "thenlper/gte-base" "nomic-ai/nomic-embed-text-v1"; do
#  for dataset in "mteb/sts12-sts" "mteb/sts13-sts" "mteb/sts14-sts" "mteb/sts15-sts" "mteb/amazon_polarity" "dennlinger/wiki-paragraphs" "mteb/banking77" "mteb/sickr-sts" "mteb/biosses-sts" "mteb/stsbenchmark-sts" "mteb/imdb" "nvidia/OpenMathInstruct-1" "snli" "Open-Orca/OpenOrca" "cnn_dailymail" "EdinburghNLP/xsum"; do
>>>>>>> 0edeba6 (merge knife estimator)
#    for split in "validation" "test"; do
#    sbatch --job-name=qa \
#      --account=ehz@v100 \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --no-requeue \
#      --cpus-per-task=10 \
#      --hint=nomultithread \
#      --time=6:00:00 \
#      --output=jobinfo_emb/testlib%j.out \
#      --error=jobinfo_emb/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python ../scripts/generate_embeddings.py \
#        --model ${model} \
#        --dataset ${dataset} \
#        --split ${split} \
#        --output_dir ${output_dir} \
<<<<<<< HEAD
#        --device cuda \
#        --batch_size 256"
#  done
#  done
#done
#
=======
#        --device cuda"
#  done
#  done
#done

>>>>>>> 0edeba6 (merge knife estimator)

#LARGE_MODELS = [
#    "WhereIsAI/UAE-Large-V1",
#    "Salesforce/SFR-Embedding-Mistral",
#    "GritLM/GritLM-7B",
#    "jspringer/echo-mistral-7b-instruct-lasttoken",
#]

<<<<<<< HEAD
for model in  "Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B"; do
  for dataset in "mteb/sts12-sts" "mteb/sts13-sts" "mteb/sts14-sts" "mteb/sts15-sts" "mteb/amazon_polarity" "mteb/banking77" "mteb/sickr-sts" "mteb/biosses-sts" "mteb/stsbenchmark-sts" "mteb/imdb"  "snli"; do
    for split in "validation" "test"; do
    sbatch --job-name=emb \
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
=======
for model in "WhereIsAI/UAE-Large-V1" "Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B" "jspringer/echo-mistral-7b-instruct-lasttoken"; do
  for dataset in "mteb/sts12-sts" "mteb/sts13-sts" "mteb/sts14-sts" "mteb/sts15-sts" "mteb/amazon_polarity" "dennlinger/wiki-paragraphs" "mteb/banking77" "mteb/sickr-sts" "mteb/biosses-sts" "mteb/stsbenchmark-sts" "mteb/imdb" "nvidia/OpenMathInstruct-1" "snli" "Open-Orca/OpenOrca" "cnn_dailymail" "EdinburghNLP/xsum"; do
    for split in "validation" "test"; do
    sbatch --job-name=emb \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=6:00:00 \
      --output=jobinfo_emb/testlib%j.out \
      --error=jobinfo_emb/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
>>>>>>> 0edeba6 (merge knife estimator)
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python ../scripts/generate_embeddings.py \
        --model ${model} \
        --dataset ${dataset} \
        --split ${split} \
        --output_dir ${output_dir} \
<<<<<<< HEAD
        --device cuda \
        --batch_size 64"
=======
        --device cuda"
>>>>>>> 0edeba6 (merge knife estimator)
  done
  done
done