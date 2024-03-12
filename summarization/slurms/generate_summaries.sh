output_dir=$1

# slauw87/bart_summarisation
# flax-community/t5-base.csv-cnn-dm
# sshleifer/distilbart-cnn-6-6
# sshleifer/distilbart-cnn-12-3
# sshleifer/distilbart-cnn-12-6
# sshleifer/distill-pegasus-xsum-16-4
# sshleifer/distill-pegasus-cnn-16-4
# sshleifer/distill-pegasus-xsum-16-8
# sshleifer/distilbart-xsum-6-6
# sshleifer/distill-pegasus-xsum-12-12
# sshleifer/distilbart-xsum-12-1
# sshleifer/distilbart-cnn-12-6
# google/pegasus-large google/pegasus-multi_news google/pegasus-arxiv facebook/bart-large-cnn
# sshleifer/distilbart-cnn-6-6 sshleifer/distilbart-cnn-12-3 sshleifer/distilbart-cnn-12-6 sshleifer/distill-pegasus-xsum-16-4 sshleifer/distill-pegasus-cnn-16-4 sshleifer/distill-pegasus-xsum-16-8 sshleifer/distilbart-xsum-6-6 sshleifer/distill-pegasus-xsum-12-12 sshleifer/distilbart-xsum-12-1 sshleifer/distilbart-cnn-12-6



# french
# "csebuetnlp/mT5_multilingual_XLSum",
# "moussaKam/barthez-orangesum-abstract",
# "plguillou/t5-base-fr-sum-cnndm",

for model in "csebuetnlp/mT5_multilingual_XLSum" "moussaKam/barthez-orangesum-abstract" "plguillou/t5-base-fr-sum-cnndm"; do
  for config in beam_sampling_50 beam_sampling_100 beam_sampling_200; do
    for ds in xlsum_fra orange_fra mlsum_fra; do
      sbatch --job-name=gensumm \
        --account=ehz@v100 \
        --gres=gpu:1 \
        --partition=gpu_p2 \
        --no-requeue \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=5:00:00 \
        --output=jobinfo_gen/testlib%j.out \
        --error=jobinfo_gen/testlib%j.err \
        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/generate_summaries.py \
        --model_name ${model} \
        --dataset_name ${ds} \
        --dataset_path data/datasets \
        --decoding_config  ${config} \
        --batch_size 16 \
        --device cuda \
        --output_dir ${output_dir} \
        --limit 10000
        "
    done
  done
done

for model in "Shahm/bart-german" "Shahm/t5-small-german" "Einmalumdiewelt/PegasusXSUM_GNAD"; do
  for config in beam_sampling_50 beam_sampling_100 beam_sampling_200; do
    for ds in xlsum_deu mlsum_deu; do
      sbatch --job-name=gensumm \
        --account=ehz@v100 \
        --gres=gpu:1 \
        --partition=gpu_p2 \
        --no-requeue \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=5:00:00 \
        --output=jobinfo_gen/testlib%j.out \
        --error=jobinfo_gen/testlib%j.err \
        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/generate_summaries.py \
        --model_name ${model} \
        --dataset_name ${ds} \
        --dataset_path data/datasets \
        --decoding_config  ${config} \
        --batch_size 16 \
        --device cuda \
        --output_dir ${output_dir} \
        --limit 10000
        "
    done
  done
done

for model in "josmunpen/mt5-small-spanish-summarization" "IIC/mt5-spanish-mlsum" "mrm8488/bert2bert_shared-spanish-finetuned-summarization" "eslamxm/mt5-base-finetuned-Spanish"; do
  for config in beam_sampling_50 beam_sampling_100 beam_sampling_200; do
    for ds in xlsum_spa mlsum_spa; do
      sbatch --job-name=gensumm \
        --account=ehz@v100 \
        --gres=gpu:1 \
        --partition=gpu_p2 \
        --no-requeue \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=5:00:00 \
        --output=jobinfo_gen/testlib%j.out \
        --error=jobinfo_gen/testlib%j.err \
        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/generate_summaries.py \
        --model_name ${model} \
        --dataset_name ${ds} \
        --dataset_path data/datasets \
        --decoding_config  ${config} \
        --batch_size 16 \
        --device cuda \
        --output_dir ${output_dir} \
        --limit 10000
        "
    done
  done
done
#
#for model in flax-community/t5-base.csv-cnn-dm  "airKlizz/mt5-base-wikinewssum-all-languages" "Falconsai/text_summarization" "Falconsai/medical_summarization" sshleifer/distilbart-cnn-6-6 sshleifer/distilbart-cnn-12-3 sshleifer/distilbart-cnn-12-6 sshleifer/distill-pegasus-xsum-16-4 sshleifer/distill-pegasus-cnn-16-4 sshleifer/distill-pegasus-xsum-16-8 sshleifer/distilbart-xsum-6-6 sshleifer/distill-pegasus-xsum-12-12 sshleifer/distilbart-xsum-12-1 sshleifer/distilbart-cnn-12-6; do
#  for config in beam_sampling_50 beam_sampling_100 beam_sampling_200; do
#    for ds in  rotten_tomatoes_long ; do # "multi_news" "rotten_tomatoes" "peer_read" "xsum" "arxiv"; do #
#      sbatch --job-name=gensumm \
#        --account=ehz@v100 \
#        --gres=gpu:1 \
#        --partition=gpu_p2 \
#        --no-requeue \
#        --cpus-per-task=10 \
#        --hint=nomultithread \
#        --time=5:00:00 \
#        --output=jobinfo_gen/testlib%j.out \
#        --error=jobinfo_gen/testlib%j.err \
#        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python EMIR/summarization/summarization_evaluation/generate_summaries.py \
#        --model_name ${model} \
#        --dataset_name ${ds} \
#        --dataset_path data/datasets \
#        --decoding_config  ${config} \
#        --batch_size 16 \
#        --device cuda \
#        --output_dir ${output_dir} \
#        --limit 10000
#        "
#    done
#  done
#done
#
#for model in sshleifer/distill-pegasus-cnn-16-4 google/pegasus-large google/pegasus-multi_news google/pegasus-arxiv facebook/bart-large-cnn; do
#  for config in 05 08 095 099; do
#    for ds in  multi_news cnn_dailymail xsum rotten_tomatoes; do # "multi_news" "rotten_tomatoes" "peer_read" "xsum" "arxiv"; do #
#      sbatch --job-name=gensumm \
#        --account=ehz@v100 \
#        --gres=gpu:1 \
#        --partition=gpu_p2 \
#        --no-requeue \
#        --cpus-per-task=10 \
#        --hint=nomultithread \
#        --time=8:00:00 \
#        --output=jobinfo_gen/testlib%j.out \
#        --error=jobinfo_gen/testlib%j.err \
#        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python EMIR/summarization/summarization_evaluation/generate_summaries.py \
#        --model_name ${model} \
#        --dataset_name ${ds} \
#        --dataset_path data/datasets \
#        --decoding_config  beam_sampling_topp_${config} \
#        --batch_size 16 \
#        --device cuda \
#        --output_dir ${output_dir} \
#        --limit 10000
#        "
#    done
#  done
#done
