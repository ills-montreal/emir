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
#        --device cuda"
#  done
#  done
#done


#LARGE_MODELS = [
#    "WhereIsAI/UAE-Large-V1",
#    "Salesforce/SFR-Embedding-Mistral",
#    "GritLM/GritLM-7B",
#    "jspringer/echo-mistral-7b-instruct-lasttoken",
#]

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
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python ../scripts/generate_embeddings.py \
        --model ${model} \
        --dataset ${dataset} \
        --split ${split} \
        --output_dir ${output_dir} \
        --device cuda"
  done
  done
done