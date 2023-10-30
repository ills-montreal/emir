
import transformers
import torch
from datasets import load_dataset

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_name", type=str, default="rotten_tomatoes")
    parser.add_argument("--decoding_config", type=str, default="top_p_sampling")

    args = parser.parse_args()

    return args



def load_dataset(dataset_name):
    if dataset_name == "rotten_tomatoes":
        pass


def main():
    pass


if __name__ == '__main__':
    main()