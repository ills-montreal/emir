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
    "mteb/banking77": None,
    "mteb/sickr-sts": None,
    "mteb/biosses-sts": None,
    "mteb/stsbenchmark-sts": None,
    "mteb/imdb": None,
    "snli": None,
    "dennlinger/wiki-paragraphs": None,
}

# TASk_DATASET : Dict[Dict[str, str]], store config and type of task
TASKS_DATASET = {
    "yelp_review_full": {
        "dataset_name": "yelp_review_full",
        "config": None,
        "task": "Sentiment classification",
        "num_classes": 5,
    },  # text, label
    "paws-x;en": {
        "dataset_name": "paws-x",
        "config": "en",
        "task": "Paraphrase identification",
        "num_classes": 2,
    },  # sentence1, sentence2, label
    "sst2": {
        "dataset_name": "sst2",
        "config": None,
        "task": "Sentiment classification",
        "num_classes": 2,
    },  # sentence, label
    "tweet_eval;emoji": {
        "dataset_name": "tweet_eval",
        "config": "emoji",
        "task": "Emoji prediction",
        "n_classes": 20,
    },  # text, label
    "tweet_eval;emotion": {
        "dataset_name": "tweet_eval",
        "config": "emotion",
        "task": "Emotion prediction",
        "n_classes": 4,
    },  # text, label
    "tweet_eval;sentiment": {
        "dataset_name": "tweet_eval",
        "config": "sentiment",
        "task": "Sentiment prediction",
        "n_classes": 3,
    },  # text, label
    "rotten_tomatoes": {
        "dataset_name": "rotten_tomatoes",
        "config": None,
        "task": "Sentiment classification",
        "num_classes": 2,
    },  # text, label
    "imdb": {
        "dataset_name": "imdb",
        "config": "plain_text",
        "task": "Sentiment classification",
        "num_classes": 2,
    },  # text, label
    "clinc_oos;plus": {
        "dataset_name": "clinc_oos",
        "config": "plus",
        "task": "Intent classification",
        "num_classes": 151,
    },  # text, intent
    "ag_news": {
        "dataset_name": "ag_news",
        "config": None,
        "task": "Topic classification",
        "num_classes": 4,
    },  # text, label
    "dair-ai/emotion": {
        "dataset_name": "dair-ai/emotion",
        "config": None,
        "task": "Emotion classification",
        "num_classes": 6,
    },  # text, label
    "banking77": {
        "dataset_name": "mteb/banking77",
        "config": None,
        "task": "Intent classification",
        "num_classes": 77,
    },  # text, label
}


def load_emd_classif_dataset(task_name):
    if task_name in TASKS_DATASET:
        dataset_name = TASKS_DATASET[task_name]["dataset_name"]
        config = TASKS_DATASET[task_name]["config"]
    else:
        raise ValueError(f"Task {task_name} not found in TASKS_DATASET")

    if config is None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, config)

    # what split is there?
    splits = list(dataset.keys())

    # normalize the datasets to have the same columns

    if task_name == "paws-x;en":
        for split in splits:
            dataset[split] = dataset[split].map(
                lambda x: {
                    "text": x["sentence1"] + "\n\n" + x["sentence2"],
                    "label": x["label"],
                }
            )

    elif task_name == "tweet_eval;emoji":
        pass
    elif task_name == "tweet_eval;emotion":
        pass
    elif task_name == "tweet_eval;sentiment":
        pass
    elif task_name == "banking77":
        pass
    elif task_name == "clinc_oos;plus":
        for split in splits:
            dataset[split] = dataset[split].map(
                lambda x: {"text": x["text"], "label": x["intent"]}
            )
    elif task_name == "rotten_tomatoes":
        pass
    elif task_name == "imdb":
        pass
    elif task_name == "yelp_review_full":
        pass
    elif task_name == "ag_news":
        pass
    elif task_name == "dair-ai/emotion":
        pass
    elif task_name == "sst2":
        for split in splits:
            dataset[split] = dataset[split].map(
                lambda x: {"text": x["sentence"], "label": x["label"]}
            )
    else:
        raise ValueError(f"Task {task_name} not found in TASKS_DATASET")

    return dataset, splits, TASKS_DATASET[task_name]


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
    dataset = dataset.head(100000)
    dataset = Dataset.from_pandas(dataset)
    # keep max 100 000 rows

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

        for task_name in TASKS_DATASET:
            print(f"Loading {task_name}...")
            dataset, _, _ = load_emd_classif_dataset(task_name)

            print(dataset)

        for dataset_name, config in AVAILABLE_DATASETS.items():
            print(f"Loading {dataset_name}...")
            if config is not None:
                load_dataset(dataset_name, config, download_mode="force_redownload")
            else:
                load_dataset(dataset_name, download_mode="force_redownload")

        print("Done!")
