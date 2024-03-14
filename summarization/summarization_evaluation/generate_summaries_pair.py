from pathlib import Path
import argparse


TEMPLATE = """
      sbatch --job-name=qa \
        --account=ehz@v100 \
        --gres=gpu:1 \
        --no-requeue \
        --cpus-per-task=10 \
        --hint=nomultithread \
        --time=5:00:00 \
        --output=jobinfo_cross/testlib%j.out \
        --error=jobinfo_cross/testlib%j.err \
        --wrap="module purge; module load pytorch-gpu/py3/2.0.0;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1; \
        python EMIR/summarization/summarization_evaluation/cross_summarizers_eval.py \
        --model paraphrase-MiniLM-L6-v2 \
        --summaries_1 ${summary_1} \
        --summaries_2 ${summary_2} \
        --output  ${output_dir} \
        --batch_size 64 \
        --n_epochs 500 \
        --stopping_criterion early_stopping \
        --eps 1e-6 \
        --lr 0.005 \
        --device cuda \
        --cond_modes 4 \
        --marg_modes 4"
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # list all csv files in the directory
    csv_files = list(args.summaries.glob("*.csv"))

    # metadata = path.split("-_-")
    # model_name, dataset_name, decoding_config, date = metadata

    # extract all metadata
    metadata = [path.stem.split("-_-") for path in csv_files]
    # model_names, dataset_names, decoding_configs, dates = zip(*metadata)


    # make pair of csv files
    pairs_csv = []
    for k, csv_1 in enumerate(csv_files):
        for csv_2 in csv_files:
            if csv_1 == csv_2:
                continue
            if not ("mistral" in str(csv_1) or "mistral" in str(csv_2)):
                continue
            pairs_csv.append((csv_1, csv_2))

    # make full path
    pairs_csv = [(csv_file_1, csv_file_2) for csv_file_1, csv_file_2 in pairs_csv]


    for csv_file_1, csv_file_2 in pairs_csv:
        print(TEMPLATE.replace("${summary_1}", str(csv_file_1)).replace("${summary_2}", str(csv_file_2)).replace("${output_dir}", args.output))


if __name__ == "__main__":
    main()