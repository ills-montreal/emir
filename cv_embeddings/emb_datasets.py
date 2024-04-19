from datasets import load_dataset


AVAILABLE_DATASETS = [
    "AI-Lab-Makerere/beans",
]

# TASk_DATASET : Dict[Dict[str, str]], store config and type of task
TASKS_DATASET = {
    "AI-Lab-Makerere/beans": {
        "dataset_name": "beans",
        "config": None,
        "task": "Image classification",
        "num_classes": 3,
    },  
}


def load_emb_dataset(dataset_name, split="test"):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    if split in dataset:
        dataset = dataset[split]
    else:
        raise ValueError(f"Split {split} not found in dataset {dataset_name}")

    return dataset


