# datasets:

<<<<<<< HEAD
<<<<<<< HEAD
output_dir=$1 # ../mis_graph/output_norma_2/mis_evaluation
embeddings_dir=$2
log_dir=$3 # ../mis_graph/output_norma_2/logs
=======
output_dir=$1
embeddings_dir=$2
log_dir=$3
>>>>>>> 33a03ab (fix kinfe estimator)
=======
output_dir=$1 # ../mis_graph/output_norma_2/mis_evaluation
embeddings_dir=$2
log_dir=$3 # ../mis_graph/output_norma_2/logs
>>>>>>> ba18e5c (working nlp embeddings + some updates to common code)

#['izhx/udever-bloom-560m',
# 'sentence-transformers/all-MiniLM-L6-v2',
# 'jinaai/jina-embedding-s-en-v1',
# 'sentence-transformers/all-distilroberta-v1',
# 'sentence-transformers/all-mpnet-base-v2',
# 'intfloat/e5-small',
# 'intfloat/multilingual-e5-small',
# 'sentence-transformers/LaBSE',
# 'BAAI/bge-base-en-v1.5',
# 'sentence-transformers/gtr-t5-xl',
# 'sentence-transformers/gtr-t5-base',
# 'avsolatorio/GIST-Embedding-v0',
# 'sentence-transformers/gtr-t5-large',
# 'sentence-transformers/sentence-t5-xl',
# 'thenlper/gte-base',
# 'sentence-transformers/sentence-t5-large',
# 'jamesgpt1/sf_model_e5',
# 'infgrad/stella-base-en-v2',
# 'thenlper/gte-large',
# 'intfloat/e5-large-v2',
# 'sentence-transformers/average_word_embeddings_komninos',
# 'sentence-transformers/average_word_embeddings_glove.6B.300d',
# 'SmartComponents/bge-micro-v2',
# 'TaylorAI/gte-tiny',
# 'sentence-transformers/msmarco-bert-co-condensor',
# 'princeton-nlp/sup-simcse-bert-base-uncased',
# 'sentence-transformers/allenai-specter',
# 'WhereIsAI/UAE-Large-V1',
# 'llmrails/ember-v1']
for model_1 in  'izhx/udever-bloom-560m' 'sentence-transformers/all-MiniLM-L6-v2' 'jinaai/jina-embedding-s-en-v1' 'sentence-transformers/all-distilroberta-v1' 'sentence-transformers/all-mpnet-base-v2' 'intfloat/e5-small' 'intfloat/multilingual-e5-small' 'sentence-transformers/LaBSE' 'BAAI/bge-base-en-v1.5' 'sentence-transformers/gtr-t5-xl' 'sentence-transformers/gtr-t5-base' 'avsolatorio/GIST-Embedding-v0' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/sentence-t5-xl' 'thenlper/gte-base' 'sentence-transformers/sentence-t5-large' 'jamesgpt1/sf_model_e5' 'infgrad/stella-base-en-v2' 'thenlper/gte-large' 'intfloat/e5-large-v2' 'sentence-transformers/average_word_embeddings_komninos' 'sentence-transformers/average_word_embeddings_glove.6B.300d' 'SmartComponents/bge-micro-v2' 'TaylorAI/gte-tiny' 'sentence-transformers/msmarco-bert-co-condensor' 'princeton-nlp/sup-simcse-bert-base-uncased' 'sentence-transformers/allenai-specter' 'WhereIsAI/UAE-Large-V1' 'llmrails/ember-v1'; do
for model_2 in  'izhx/udever-bloom-560m' 'sentence-transformers/all-MiniLM-L6-v2' 'jinaai/jina-embedding-s-en-v1' 'sentence-transformers/all-distilroberta-v1' 'sentence-transformers/all-mpnet-base-v2' 'intfloat/e5-small' 'intfloat/multilingual-e5-small' 'sentence-transformers/LaBSE' 'BAAI/bge-base-en-v1.5' 'sentence-transformers/gtr-t5-xl' 'sentence-transformers/gtr-t5-base' 'avsolatorio/GIST-Embedding-v0' 'sentence-transformers/gtr-t5-large' 'sentence-transformers/sentence-t5-xl' 'thenlper/gte-base' 'sentence-transformers/sentence-t5-large' 'jamesgpt1/sf_model_e5' 'infgrad/stella-base-en-v2' 'thenlper/gte-large' 'intfloat/e5-large-v2' 'sentence-transformers/average_word_embeddings_komninos' 'sentence-transformers/average_word_embeddings_glove.6B.300d' 'SmartComponents/bge-micro-v2' 'TaylorAI/gte-tiny' 'sentence-transformers/msmarco-bert-co-condensor' 'princeton-nlp/sup-simcse-bert-base-uncased' 'sentence-transformers/allenai-specter' 'WhereIsAI/UAE-Large-V1' 'llmrails/ember-v1'; do
    sbatch --job-name=mis \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --no-requeue \
      --cpus-per-task=10 \
      --hint=nomultithread \
      --time=5:00:00 \
      --output=jobinfo_mis/testlib%j.out \
      --error=jobinfo_mis/testlib%j.err \
<<<<<<< HEAD
<<<<<<< HEAD
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        export WANDB_MODE=offline;
=======
      --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
>>>>>>> 33a03ab (fix kinfe estimator)
=======
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        export WANDB_MODE=offline;
>>>>>>> ba18e5c (working nlp embeddings + some updates to common code)
        python ../scripts/evaluate_mis.py \
        --model_1 ${model_1} \
        --model_2 ${model_2} \
        --output_dir ${output_dir} \
        --embeddings_dir ${embeddings_dir} \
        --log_dir ${log_dir} \
        --device cuda \
<<<<<<< HEAD
<<<<<<< HEAD
        --batch_size 128  \
        --cond_modes 8 \
        --marg_modes 8 \
        --ff_layers 2 \
        --eval_batch_size 1024 \
        --n_epochs 10 \
        --n_epochs_marg 10 \
        --lr 0.001 \
        --margin_lr 0.0001 \
        --normalize_embeddings \
        --ff_layer_norm \
        "
  done
done

# cmd example
# python ../scripts/evaluate_mis.py --model_1 'sentence-transformers/all-MiniLM-L6-v2' --model_2 'sentence-transformers/all-MiniLM-L6-v2' --output_dir ../mis_graph/output_norma_2/mis_evaluation --embeddings_dir ../mis_graph/output_norma_2/embeddings --log_dir ../mis_graph/output_norma_2/logs --device cuda --batch_size 128  --cond_modes 8 --marg_modes 8 --ff_layers 2 --eval_batch_size 1024 --n_epochs 5 --n_epochs_marg 10 --lr 0.001 --margin_lr 0.001 --normalize_embeddings --ff_layer_norm
#
=======
        --batch_size 64  \
        --cond_modes 16 \
        --marg_modes 16 \
=======
        --batch_size 128  \
        --cond_modes 8 \
        --marg_modes 8 \
>>>>>>> ba18e5c (working nlp embeddings + some updates to common code)
        --ff_layers 2 \
        --eval_batch_size 1024 \
        --n_epochs 10 \
        --n_epochs_marg 10 \
        --lr 0.001 \
        --margin_lr 0.0001 \
        --normalize_embeddings \
        --ff_layer_norm \
        "
  done
done
<<<<<<< HEAD
#BAAI/bge-base-en-v1.5 infgrad/stella-base-en-v2 intfloat/e5-large-v2 intfloat/multilingual-e5-small sentence-transformers/sentence-t5-xl sentence-transformers/sentence-t5-large SmartComponents/bge-micro-v2 sentence-transformers/allenai-specter sentence-transformers/average_word_embeddings_glove.6B.300d sentence-transformers/average_word_embeddings_komninos sentence-transformers/LaBSE avsolatorio/GIST-Embedding-v0 Muennighoff/SGPT-125M-weightedmean-nli-bitfit princeton-nlp/sup-simcse-bert-base-uncased jinaai/jina-embedding-s-en-v1 sentence-transformers/msmarco-bert-co-condensor sentence-transformers/gtr-t5-base izhx/udever-bloom-560m llmrails/ember-v1 jamesgpt1/sf_model_e5 thenlper/gte-large TaylorAI/gte-tiny sentence-transformers/gtr-t5-xl intfloat/e5-small sentence-transformers/gtr-t5-large thenlper/gte-base nomic-ai/nomic-embed-text-v1 sentence-transformers/all-distilroberta-v1 sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-mpnet-base-v2 "WhereIsAI/UAE-Large-V1" "Salesforce/SFR-Embedding-Mistral" "GritLM/GritLM-7B" "jspringer/echo-mistral-7b-instruct-lasttoken"; do
>>>>>>> 33a03ab (fix kinfe estimator)
=======

# cmd example
# python ../scripts/evaluate_mis.py --model_1 'sentence-transformers/all-MiniLM-L6-v2' --model_2 'sentence-transformers/all-MiniLM-L6-v2' --output_dir ../mis_graph/output_norma_2/mis_evaluation --embeddings_dir ../mis_graph/output_norma_2/embeddings --log_dir ../mis_graph/output_norma_2/logs --device cuda --batch_size 128  --cond_modes 8 --marg_modes 8 --ff_layers 2 --eval_batch_size 1024 --n_epochs 5 --n_epochs_marg 10 --lr 0.001 --margin_lr 0.001 --normalize_embeddings --ff_layer_norm
#
>>>>>>> ba18e5c (working nlp embeddings + some updates to common code)
