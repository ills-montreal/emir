
BASE_DIR=$1

python EMIR/summarization/summarization_evaluation/merge_df.py output/metric_evaluation/${BASE_DIR}_bert/ output/${BASE_DIR}_merged/bert.csv
python EMIR/summarization/summarization_evaluation/merge_df.py output/metric_evaluation/${BASE_DIR}_classifiers/ output/${BASE_DIR}_merged/classifiers.csv
python EMIR/summarization/summarization_evaluation/merge_df.py output/metric_evaluation/${BASE_DIR}_common/ output/${BASE_DIR}_merged/common.csv
python EMIR/summarization/summarization_evaluation/merge_df.py output/metric_evaluation/${BASE_DIR}_emb/ output/${BASE_DIR}_merged/emb.csv
python EMIR/summarization/summarization_evaluation/merge_df.py output/metric_evaluation/${BASE_DIR}_mis2/ output/${BASE_DIR}_merged/mis2.csv
python EMIR/summarization/summarization_evaluation/merge_df.py output/metric_evaluation/${BASE_DIR}_shm/ output/${BASE_DIR}_merged/shm.csv



