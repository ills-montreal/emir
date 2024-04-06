import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_table_model_summary(df):
    def rename_cols(x):
        if "SHMetric" in x:
            return x.split("/")[1]
        elif "rougeLsum" in x:
            return "\\texttt{ROUGE-L}"
        elif "BARTScore" in x:
            return "\\texttt{BARTScore}"
        elif "BERTScore" in x:
            return "\\texttt{BERTScore}"
        elif "common" in x or "metadata" in x:
            return x.split("/")[1]
        else:
            return x

    df = df.copy()

    datasets = ["xsum", "cnn_dailymail", "multi_news"]
    df = df[df["metadata/Dataset name"].isin(datasets)]

    avg = df.groupby(["metadata/Model name", "metadata/Dataset name"]).mean()
    SHM = [c for c in df.columns if "SHMetric" in c and "proba_1" in c]
    # SHM = [c for c in SHM if "Attribution" in c or "Main ideas" in c]

    cols = (
        ["metadata/#params", "common/rougeLsum", "common/BERTScore", "common/BARTScore"]
        + SHM
        + ["I(summary -> text)", "H(text|summary)", "H(summary|text)"]
    )

    avg = avg[cols]

    avg["Size"] = avg["metadata/#params"].apply(lambda x: f"{int(x) // 10 ** 6} M")

    avg = avg.reset_index()

    avg.columns = [rename_cols(c) for c in avg.columns]
    # rename I(summary -> text)
    avg = avg.rename(
        columns={
            "I(summary -> text)": "$I(T,S)$",
            "Model name": "Model",
            "Dataset name": "Dataset",
            "Main ideas": "M. I.",
            "Attribution": "Attr.",
            "Repetition": "Rep.",
            "Comprehensible": "Compr.",
            "Conciseness": "Conc.",
            "Grammar": "Gram.",
            "H(text|summary)": "H(T|S)",
            "H(summary|text)": "H(S|T)",
        }
    )
    # rename dataset name and model name columns

    # sort by Size
    avg = avg.sort_values(by=["#params"])

    avg = avg.drop(columns=["#params"])
    # avg['Model'] = avg['Model'].apply(lambda x: " ".join(x.split('_')[1:]))

    avg = avg.set_index(["Model", "Dataset"])

    avg = avg[
        [
            "Size",
            "\\texttt{ROUGE-L}",
            "\\texttt{BERTScore}",
            "\\texttt{BARTScore}",
            "M. I.",
            "Attr.",
            "Rep.",
            "Compr.",
            "Conc.",
            "Gram.",
            "$I(T,S)$",
            "H(T|S)",
            "H(S|T)",
        ]
    ]

    return avg


def make_correlation_table(df):
    df = df.copy()

    df = df[~df["metadata/Decoding config"].str.contains("short")]
    # df = df[df['metadata/Decoding config'].str.contains("100") | df['metadata/Decoding config'].str.contains("50")]

    ROUGES = [
        "common/rougeLsum",
        "common/BERTScore",
        "common/BERTScore Precision",
        "common/BERTScore Recall",
        "common/BARTScore",
        "common/BLANC",
        "common/SMART1",
        "common/SMART2",
        "common/SMARTL",
    ]
    MI = ["I(summary -> text)", "I(text -> summary)"]

    SHM = [c for c in df.columns if "SHMetric" in c and "proba_1" in c]
    SHM_interesting = [c for c in SHM if "Attribution" in c or "Main ideas" in c]
    embeddings = [c for c in df.columns if "embedding" in c]

    map_tasks = {
        "mrm8488_distilroberta-finetuned-financial-news-sentiment-analysis": "Sentiment analysis",
        "roberta-base-openai-detector": "GPT detector",
        "manifesto-project_manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1": "Policy classification",
        # "jonaskoenig_topic_classification_04" : "Topic classification",
        "SamLowe_roberta-base-go_emotions": "Emotion classification",
    }

    embedding_map = {"paraphrase-MiniLM-L6-v2": "Emb. Paraphrase"}

    classification_tasks = [c + "/proba_of_error" for c in map_tasks.keys()]

    # make proba_of_error proba_of_success
    df[classification_tasks] = 1 - df[classification_tasks]
    # rename
    df = df.rename(
        columns={
            c + "/proba_of_error": c + "/proba_of_success" for c in map_tasks.keys()
        }
    )

    classification_tasks = [c + "/proba_of_success" for c in map_tasks.keys()]

    embedding_tasks = [c for c in df.columns if "embedding" in c and "dot" in c]

    df = df[~df["metadata/Decoding config"].str.contains("short")]

    df = df[
        ~df["metadata/Decoding config"].isin(
            [f"beam_sampling_{k}" for k in [5, 10, 20, 50]]
        )
    ]

    datasets = set(df["metadata/Dataset name"].dropna().unique())
    datasets -= set(["peer_read", "arxiv", "rotten_tomatoes"])

    # create a dataframe with the correlation between MI and ROUGE and the SHmetrics, grouped by dataset
    df_corr = pd.DataFrame(columns=["Dataset name", "Metric", "Correlation"])
    for dataset in datasets:
        # select dataset
        df_dataset = df[df["metadata/Dataset name"] == dataset]
        df_dataset = df_dataset[
            ROUGES + SHM + MI + classification_tasks + embedding_tasks
        ].corr("kendall")
        # add dataset name
        df_dataset["Dataset name"] = dataset

        # add metric name
        df_dataset["Metric"] = df_dataset.index

        # melt dataframe
        df_dataset = df_dataset.melt(
            id_vars=["Dataset name", "Metric"],
            var_name="Correlation",
            value_name="Value",
        )

        # append to main dataframe
        df_corr = df_corr.append(df_dataset)

    def rename_metrics(x):
        splits = x.split("/")

        if len(splits) == 1:
            if splits[0] == "I(summary -> text)":
                return "$I(S;T)$"
            else:
                return x
        else:
            if splits[0] in map_tasks.keys():
                return map_tasks[splits[0]]
            # if splits[1] in embedding_map.keys():
            # return embedding_map[splits[0]]
            else:
                if splits[1] == "rougeLsum":
                    return "\\texttt{ROUGE-L}"
                elif splits[1] == "BERTScore":
                    return "\\texttt{BERTScore}"
                elif splits[1] == "BARTScore":
                    return "\\texttt{BARTScore}"
                if "embedding" in x:
                    if splits[1]:
                        return embedding_map[splits[1]]
                else:
                    return splits[1]

    df_corr = df_corr.pivot(
        index=["Dataset name", "Metric"], columns="Correlation", values="Value"
    )

    # Keep shmetric only in columns
    df_corr = pd.concat(
        {
            "SH.": df_corr[[c for c in df_corr.columns if "SHMetric" in c]],
            "CT.": df_corr[classification_tasks],
            "Emb.": df_corr[embedding_tasks],
            "Common": df_corr[ROUGES],
        },
        axis=1,
    )

    idx = pd.IndexSlice

    # Select index to be displayed
    df_corr = df_corr.loc[
        idx[
            :,
            [
                "I(summary -> text)",
            ]
            + ROUGES
            + SHM_interesting,
        ],
        :,
    ]
    # df_corr = df_corr.dropna()

    # rename columns
    df_corr.columns = pd.MultiIndex.from_tuples(
        [(c[0].replace("_", "-"), rename_metrics(c[1])) for c in df_corr.columns]
    )

    df_corr = df_corr.reset_index()
    # rename Metric
    df_corr[("Metric", "")] = df_corr[("Metric", "")].apply(rename_metrics)

    df_corr = df_corr.set_index(["Dataset name", "Metric"])
    df_corr = df_corr.sort_index()

    # Remove "_" from column names

    return df_corr
