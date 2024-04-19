import argparse
from pathlib import Path
import torch
import numpy as np
from cache_models import MODELS
from embed import extract_embedds
from emb_datasets import (
    load_emb_dataset,
    AVAILABLE_DATASETS,
    TASKS_DATASET,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification_task", action="store_true", default=False)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
    )
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224-in21k")
    # parser.add_argument("--dataset", type=str, default="AI-Lab-Makerere/beans")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def main():
    args = parse_args()
    
    for dataset_name in AVAILABLE_DATASETS:
        for split in ["train", "validation", "test"]:

            print("START")
            # print("CLASSIFICATION TASK")

            print("LOADING DATASET ", dataset_name, split)
            dataset = load_emb_dataset(dataset_name=dataset_name, split=split)
            print("DATASET LOADED ", dataset_name, split)

            print("EXTRACTING EMBEDDINGS")
            embeddeds = extract_embedds(args.model, dataset, batch_size=args.batch_size, device=args.device)
            print("EXTRACTING DONE")

            embeddeds_tensor = torch.from_numpy(np.asarray([embeddeds[idx]['embeddings'] for idx in range(len(embeddeds))]))
            labels_tensor = torch.from_numpy(np.asarray([embeddeds[idx]['labels'] for idx in range(len(embeddeds))]))
            
            print(embeddeds_tensor.shape)
            print(labels_tensor.shape)

            _output_dir = Path(args.output_dir) / args.model / dataset_name / split
            _output_dir.mkdir(parents=True, exist_ok=True)

            torch.save(embeddeds_tensor, _output_dir / "embeddings.pt")
            torch.save(labels_tensor, _output_dir / "labels.pt")


if __name__ == "__main__":
    import sys

    print(sys.version)
    main()



























# MODELS = [
#     'google/vit-base-patch16-224-in21k', 
#     'google/vit-base-patch16-224',
#     'google/vit-base-patch32-384',
# ]

# feature_extractor = ViTFeatureExtractor.from_pretrained(MODELS[0])



# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from PIL import Image
# import requests
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# feature_extractor = ViTFeatureExtractor.from_pretrained(MODELS[0])
# model = ViTForImageClassification.from_pretrained(MODELS[0])
# inputs = feature_extractor(images=image, return_tensors="pt")


# print(inputs['pixel_values'].shape)
# print(inputs['pixel_values'].reshape(-1).shape)

# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])



# # Load model directly
# from transformers import AutoImageProcessor, AutoModelForImageClassification

# processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
# model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")



# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# processor = ViTImageProcessor.from_pretrained(MODELS[0])
# model = ViTForImageClassification.from_pretrained(MODELS[0])

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])


# from datasets import load_dataset

# # load cifar10 (only small portion for demonstration purposes) 
# train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
# # split up training into training + validation
# splits = train_ds.train_test_split(test_size=0.1)
# train_ds = splits['train']
# val_ds = splits['test']



# id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
# label2id = {label:id for id,label in id2label.items()}
# id2label







