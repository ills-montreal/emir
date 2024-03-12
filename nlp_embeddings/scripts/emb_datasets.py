from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from sys import argv


AVAILABLE_DATASETS = {
    "mteb/sts12-sts": None,
    "mteb/sts13-sts": None,
    "mteb/sts14-sts": None,
    "mteb/sts15-sts": None,
    "mteb/amazon_polarity": None,
    "dennlinger/wiki-paragraphs": None,
    "mteb/banking77": None,
    "mteb/sickr-sts": None,
    "mteb/biosses-sts": None,
    "mteb/stsbenchmark-sts": None,
    "mteb/imdb": None,
    "nvidia/OpenMathInstruct-1": None,
    "snli": None,
    "Open-Orca/OpenOrca": None,
    "cnn_dailymail": "3.0.0",
    "EdinburghNLP/xsum": None,
}


def load_emb_dataset(dataset_name, config, split="test"):
    # Load the dataset

    if config is None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, config)

    if split in dataset:
        dataset = dataset[split]
    else:
        raise ValueError(f"Split {split} not found in dataset {dataset_name}")

    # to pandas
    dataset = dataset.to_pandas()

    datasets = []
    d_texts = None

    print("Dataset name: ", dataset_name)
    print("Conf: ", config)
    print("Columns: ", dataset.columns)

    if "text" in dataset.columns:
        datasets.append(dataset.copy()[["text"]])

    if "sentence1" in dataset.columns and "sentence2" in dataset.columns:
        sentence1 = dataset["sentence1"]
        sentence2 = dataset["sentence2"]

        # rename to text
        sentence1 = sentence1.rename("text").to_frame()
        sentence2 = sentence2.rename("text").to_frame()

        datasets.append(sentence1)
        datasets.append(sentence2)

    if "article" in dataset.columns and "highlights" in dataset.columns:
        article = dataset["article"]
        highlights = dataset["highlights"]

        # rename to text
        article = article.rename("text").to_frame()
        highlights = highlights.rename("text").to_frame()

        datasets.append(article)
        datasets.append(highlights)

    if "premise" in dataset.columns and "hypothesis" in dataset.columns:
        premise = dataset["premise"]
        hypothesis = dataset["hypothesis"]

        # rename to text
        premise = premise.rename("text").to_frame()
        hypothesis = hypothesis.rename("text").to_frame()

        datasets.append(premise)
        datasets.append(hypothesis)

    if "question" in dataset.columns and "response" in dataset.columns:
        question = dataset["question"]
        response = dataset["response"]

        # rename to text
        question = question.rename("text").to_frame()
        response = response.rename("text").to_frame()

        datasets.append(question)
        datasets.append(response)

    if "question" in dataset.columns and "context" in dataset.columns:
        question = dataset["question"]

        question = question.rename("text").to_frame()

        datasets.append(question)

    if len(datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_name} does not contain any text columns ({dataset.columns})"
        )

    dataset = pd.concat(datasets, ignore_index=True)
    dataset = dataset.drop_duplicates()

    # make a dataset from the pandas dataframe
    dataset = Dataset.from_pandas(dataset)

    return dataset


if __name__ == "__main__":
    if len(argv) > 1 and argv[1] == "check":
        for dataset_name, config in AVAILABLE_DATASETS.items():
            try:
                load_emb_dataset(dataset_name, config, split="test")
            except Exception as e:
                print(f"Error in {dataset_name} / test: {e}")

            try:
                load_emb_dataset(dataset_name, config, split="train")
            except Exception as e:
                print(f"Error in {dataset_name} / train: {e}")

            try:
                load_emb_dataset(dataset_name, config, split="validation")
            except Exception as e:
                print(f"Error in {dataset_name} / validation: {e}")

    else:
        print("Caching datasets...")
        for dataset_name, config in AVAILABLE_DATASETS.items():
            print(f"Loading {dataset_name}...")
            if config is not None:
                load_dataset(dataset_name, config, download_mode="force_redownload")
            else:
                load_dataset(dataset_name, download_mode="force_redownload")

        print("Done!")


instru = "Do that"
answer = "I did it"
template = """<s> [INST] [INSTR] [/INST] [ANSW] </s>"""

template = template.replace("[INST]", "INST")
template = template.replace("[INSTR]", instru)
