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

for model in "mistralai/Mistral-7B-Instruct-v0.2"; do
  for config in beam_sampling_200; do
    for ds in  multi_news cnn_dailymail xsum rotten_tomatoes; do # "multi_news" "rotten_tomatoes" "peer_read" "xsum" "arxiv"; do #
      sbatch --job-name=gensumm \
        --account=ehz@v100 \
        --gres=gpu:1 \
        --partition=gpu_p2 \
        --no-requeue \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=18:00:00 \
        --output=jobinfo_gen/testlib%j.out \
        --error=jobinfo_gen/testlib%j.err \
        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python EMIR/summarization/summarization_evaluation/generate_summaries_decoder.py \
        --model_name ${model} \
        --dataset_name ${ds} \
        --dataset_path data/datasets \
        --decoding_config  ${config} \
        --batch_size 4 \
        --device cuda \
        --output_dir ${output_dir} \
        --limit 10000
        "
    done
  done
done
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
