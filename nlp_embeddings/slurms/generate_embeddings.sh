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

#    "croissantllm/base_5k",
#    "croissantllm/base_50k",
#    "croissantllm/base_100k",
#    "croissantllm/base_150k",
#    "croissantllm/CroissantCool",
#    "HuggingFaceM4/tiny-random-LlamaForCausalLM",
#    "croissantllm/CroissantLLMBase",
#    "google/gemma-2b",
#    "google/gemma-2b-it",

#for model in "croissantllm/base_5k" "croissantllm/base_50k" "croissantllm/base_100k" "croissantllm/base_150k" "croissantllm/CroissantCool" "HuggingFaceM4/tiny-random-LlamaForCausalLM" "croissantllm/CroissantLLMBase" "google/gemma-2b" "google/gemma-2b-it"; do
#  for dataset in "mteb/sts12-sts" "mteb/sts13-sts" "mteb/sts14-sts" "mteb/sts15-sts" "mteb/amazon_polarity" "dennlinger/wiki-paragraphs" "mteb/banking77" "mteb/sickr-sts" "mteb/biosses-sts" "mteb/stsbenchmark-sts" "mteb/imdb" "snli"; do
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
#        --batch_si 32"
#  done
#  done
#done
#

#LARGE_MODELS = [
#    "WhereIsAI/UAE-Large-V1",
#    "Salesforce/SFR-Embedding-Mistral",
#    "GritLM/GritLM-7B",
#    "jspringer/echo-mistral-7b-instruct-lasttoken",
#]

# "mteb/sts12-sts" "mteb/sts13-sts" "mteb/sts14-sts" "mteb/sts15-sts" "mteb/amazon_polarity" "mteb/banking77" "mteb/sickr-sts" "mteb/biosses-sts" "mteb/stsbenchmark-sts" "mteb/imdb"  "snli"; do


#"Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B" "jspringer/echo-mistral-7b-instruct-lasttoken"; do


#    "NousResearch/Llama-2-7b-hf",
#    "togethercomputer/LLaMA-2-7B-32K",
#    "google/gemma-7b",
#    "google/gemma-7b-it",

#for model in  "NousResearch/Llama-2-7b-hf" "togethercomputer/LLaMA-2-7B-32K" "google/gemma-7b" "google/gemma-7b-it"; do


# to rerun
#../output/izhx/udever-bloom-560m
#../output/sentence-transformers/gtr-t5-xl
#jspringer/echo-mistral-7b-instruct-lasttoken
#../output/NousResearch/Llama-2-7b-hf
#../output/togethercomputer/LLaMA-2-7B-32K
#../output/google/gemma-7b
#../output/google/gemma-7b-it
#../output/google/gemma-7b
#../output/google/gemma-7b-it
# ../output/Salesforce/SFR-Embedding-Mistral
# ../output/GritLM/GritLM-7B


for model in izhx/udever-bloom-560m sentence-transformers/gtr-t5-xl jspringer/echo-mistral-7b-instruct-lasttoken NousResearch/Llama-2-7b-hf togethercomputer/LLaMA-2-7B-32K google/gemma-7b google/gemma-7b-it google/gemma-7b google/gemma-7b-it Salesforce/SFR-Embedding-Mistral GritLM/GritLM-7B; do
  for dataset in  "mteb/sts12-sts" "mteb/sts13-sts" "mteb/sts14-sts" "mteb/sts15-sts" "mteb/amazon_polarity" "dennlinger/wiki-paragraphs" "mteb/banking77" "mteb/sickr-sts" "mteb/biosses-sts" "mteb/stsbenchmark-sts" "mteb/imdb" "snli"; do
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
      --wrap="module purge; module load cpuarch/amd ; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python ../scripts/generate_embeddings.py \
        --model ${model} \
        --dataset ${dataset} \
        --split ${split} \
        --output_dir ${output_dir} \
        --batch_size 32"
  done
  done
done