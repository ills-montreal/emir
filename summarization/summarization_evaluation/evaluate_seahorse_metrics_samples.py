import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

map_questionnumber_to_question = {
    "question1": "SHMetric/Comprehensible",
    "question2": "SHMetric/Repetition",
    "question3": "SHMetric/Grammar",
    "question4": "SHMetric/Attribution",
    "question5": "SHMetric/Main ideas",
    "question6": "SHMetric/Conciseness",
}

def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")

# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        default="repetition",
    )
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--select", type=str, default="*")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")


    args = parser.parse_args()
    return args

def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file

    df = pd.read_csv(path).dropna()

    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["text", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df


def evaluate_classification_task(model, tokenizer, question, df, batch_size):

    texts = df.text.tolist()
    summaries = df.summary.tolist()

    template = "premise: {premise} hypothesis: {hypothesis}"
    ds = [template.format(premise=text[:20*1024], hypothesis=summary) for text, summary in zip(texts, summaries)]


    eval_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    metrics = {f"{question}/proba_1": [], f"{question}/proba_0": [], f"{question}/guess": []}

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            # tokenize the batch
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            # move the inputs to the device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            N_inputs = inputs["input_ids"].shape[0]
            # make decoder inputs to be <pad>
            decoder_input_ids = torch.full((N_inputs, 1), tokenizer.pad_token_id, dtype=torch.long, device=model.device)

            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            # retrieve logits for the last token and the scores for 0 and 1
            logits = logits[:, -1, [497, 333]]

            # compute the probabilities
            probs = F.softmax(logits, dim=-1)

            # compute the guess
            guess = probs.argmax(dim=-1)

            # append the metrics
            metrics[f"{question}/proba_1"].extend(probs[:, 1].tolist())
            metrics[f"{question}/proba_0"].extend(probs[:, 0].tolist())
            metrics[f"{question}/guess"].extend(guess.tolist())

    # average the metrics

    # metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    return metrics

def main():
    args = parse_args()

    model_name = f"google/seahorse-large-q{args.question}"
    question = map_questionnumber_to_question[f"question{args.question}"]

    # load the model
    # load in float16 to save memory
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = parse_summaries(args.summaries)

    metrics = evaluate_classification_task(model, tokenizer, question, df, args.batch_size)

    # make a dataframe with the metric
    df_metrics = pd.DataFrame(metrics)

    # merge the metrics with the summaries
    df = parse_summaries(args.summaries)
    df = pd.concat([df, df_metrics], axis=1)

    # save the dataframe
    df.to_csv(args.summaries)




if __name__ == "__main__":
    main()
