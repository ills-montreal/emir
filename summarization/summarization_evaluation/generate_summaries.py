import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime

import transformers

GENERATION_CONFIGS = {
    "top_p_sampling": {
        "max_new_tokens": 100,
        "do_sample": True,
        "top_p": 0.95,
    }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_name", type=str, default="rotten_tomatoes")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--decoding_config", type=str, default="top_p_sampling")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_dir", type=str, default="output")

    # limit the number of samples to generate
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    return args

# write command example
# python EMIR/summarization/summarization_evaluation/generate_summaries.sh.py --model_name facebook/bart-large-cnn --dataset_name rotten_tomatoes --dataset_path data/datasets --decoding_config top_p_sampling --batch_size 16 --device cuda --output_dir output


def load_dataset(dataset_name, dataset_path=None) -> Dataset:
    if dataset_name == "rotten_tomatoes":
        path = (
            Path(dataset_path)
            / "rotten_tomatoes_critic_reviews.csv"
        )
        dataset = pd.read_csv(path)

        # get rotten_tomatoes_link as id, review_content as text and review_score as sentiment
        new_dataset = pd.DataFrame(
            {
                "id": dataset.rotten_tomatoes_link,
                "text": dataset.review_content,
                "sentiment": dataset.review_score,
            }
        )

        # remove rows with missing values
        new_dataset = new_dataset.dropna()

        dataset = new_dataset

    elif dataset_name == "peer_read":
        path = Path(dataset_path) / "peer_read.csv"
        dataset = pd.read_csv(path)

        # dataframe has columns ['id', 'review_idx',  'title', 'abstract', 'review']
        # rename review to text
        dataset = dataset.rename(columns={"review": "text"})

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # make a dataset from the dataframe
    dataset = Dataset.from_pandas(dataset)

    return dataset


def evaluate_summarizer(
    model, tokenizer, dataset: Dataset, decoding_config, batch_size: int
) -> Dataset:
    """
    @param model: The model used to generate the summaries
    @param tokenizer: The tokenizer used to tokenize the text and the summary
    @param dataset: A dataset with the text
    @param decoding_config: Dictoionary with the decoding config
    @param batch_size: The batch size used to generate the summaries
    @return: The same dataset with the summaries added
    """
    # create a dataset with the text and the summary

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # generate summaries
    summaries = []
    print("Generating summaries...")
    for batch in tqdm(dataloader):
        text = batch["text"]
        inputs = tokenizer(
            text,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # move inputs to device
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        # generate summaries
        outputs = model.generate(
            **inputs,
            **decoding_config,
        )

        # decode summaries
        summaries.extend(
            [
                tokenizer.decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for output in outputs
            ]
        )

    # add summaries to the huggingface dataset
    dataset = dataset.map(lambda example: {"summary": summaries.pop(0)})

    return dataset


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")


def main():
    args = parse_args()

    # load the model
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    # move model to device
    model = model.to(args.device)


    # load the dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_path)

    # limit the number of samples
    if args.limit is not None:
        _lim = min(args.limit, len(dataset))
        dataset = dataset.select(range(_lim))

    # generate summaries
    dataset = evaluate_summarizer(
        model,
        tokenizer,
        dataset,
        GENERATION_CONFIGS[args.decoding_config],
        args.batch_size,
    )

    # save the dataset
    # add unique date in name
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_name = sanitize_model_name(args.model_name)
    output_path = (
        Path(args.output_dir)
        / f"{model_name}_{args.dataset_name}_{args.decoding_config}_{date}.csv"
    )

    # create output dir if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    dataset.to_pandas().to_csv(output_path, index=False, encoding="utf-8", sep=";")


if __name__ == "__main__":
    main()
