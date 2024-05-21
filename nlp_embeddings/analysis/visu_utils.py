from netgraph import Graph, InteractiveGraph
import networkx as nx
import numpy as np
import seaborn as sns
from networkx.algorithms.community import (
    girvan_newman,
    modularity_max,
    louvain_communities,
)
import scipy
from sklearn.metrics import mean_squared_error

import pandas as pd

from matplotlib import patheffects
import matplotlib.pyplot as plt


def text_embeddings_metadata_results_preprocessing(df, short_names=False, models=None):
    df["I(X_1->X_2)/H(X_2)"] = df["I(X_1->X_2)"] / df["H(X_2)"]
    df["I(X_1->X_2)/d_1"] = df["I(X_1->X_2)"] / df["d_1"]
    df["I(X_1->X_2)/d_2"] = df["I(X_1->X_2)"] / df["d_2"]

    df["model_1"] = df["model_1"].apply(lambda x: "/".join(x.split("/")[-2:]))
    df["model_2"] = df["model_2"].apply(lambda x: "/".join(x.split("/")[-2:]))

    df["datasets"] = df["datasets"].apply(lambda x: str(tuple(sorted(list(eval(x))))))

    df = (
        df.groupby(["model_1", "model_2", "marg_modes", "cond_modes", "datasets"])
        .first()
        .reset_index()
    )

    # experiments crashed on these models
    df = df[~(df["model_1"] == "jinaai/jina-embedding-s-en-v1")]
    df = df[~(df["model_2"] == "jinaai/jina-embedding-s-en-v1")]

    df = df[df["model_1"].isin(models) & df["model_2"].isin(models)]

    if short_names:
        df["model_1"] = df["model_1"].apply(lambda x: x.split("/")[-1])
        df["model_2"] = df["model_2"].apply(lambda x: x.split("/")[-1])

    # add philips names

    df["X"] = df["model_1"]
    df["Y"] = df["model_2"]

    return df


def make_table_pivot(metric, df):
    table = df
    table = (
        table[["model_1", "model_2", metric]]
        .pivot(columns="model_1", index="model_2", values=metric)
        .fillna(0)
    )

    return table


def make_rankings_table(
    classification_df, df, metric, filter_same_models=True, aggrefate_tasks=True, q=0.5
):
    ordered_classif_df = (
        classification_df.sort_values("success", ascending=False)
        .groupby(["model", "dataset"])
        .first()
        .reset_index()
    )

    ranks = (
        ordered_classif_df.groupby("dataset")["success"]
        .rank(ascending=False)
        .reset_index()
    )
    ordered_classif_df["rank"] = ranks["success"]

    if aggrefate_tasks:
        ordered_classif_df = (
            ordered_classif_df.groupby("model")[["success", "rank"]]
            .mean()
            .sort_values("success", ascending=False)
        )

    # rename success to value
    classification_ranking = ordered_classif_df.rename(columns={"success": "value"})
    classification_ranking = classification_ranking.reset_index()
    classification_ranking["ranking"] = "classification"

    table = make_table_pivot(metric, df)

    informativeness_ranking = (
        table.quantile(q, axis=0)
        .sort_values(ascending=False)
        .to_frame()
        .rename(columns={q: 0})
    )

    # rename 0 to informativeness
    informativeness_ranking = informativeness_ranking.rename(columns={0: "value"})

    informativeness_ranking["ranking"] = "informativeness"
    informativeness_ranking["rank"] = informativeness_ranking["value"].rank(
        ascending=False
    )
    informativeness_ranking = informativeness_ranking.reset_index()

    informativeness_ranking = informativeness_ranking.rename(
        columns={"index": "model", "model_1": "model"}
    )

    if filter_same_models:
        common_models = set(classification_ranking["model"]).intersection(
            set(informativeness_ranking["model"])
        )

        classification_ranking = classification_ranking[
            classification_ranking["model"].isin(common_models)
        ]
        informativeness_ranking = informativeness_ranking[
            informativeness_ranking["model"].isin(common_models)
        ]

    G = nx.from_pandas_adjacency(table, create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))

    communities = louvain_communities(G, resolution=1.1)
    communities = list(communities)

    ranking = pd.concat([classification_ranking, informativeness_ranking], axis=0)

    # model to community
    node_to_community = {
        node: i for i, community in enumerate(communities) for node in community
    }

    ranking["community"] = ranking["model"].apply(
        lambda x: node_to_community[x] if x in node_to_community else -1
    )

    return ranking


def make_communities_from_table(table):
    G = nx.from_pandas_adjacency(table, create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))

    communities = louvain_communities(G, resolution=1.1)
    communities = list(communities)

    node_to_community = {
        node: i for i, community in enumerate(communities) for node in community
    }

    return node_to_community


def annotate_regplot(ax, data, x, y):
    slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(
        x=data[x], y=data[y]
    )
    rmse = mean_squared_error(data[x], data[y], squared=False)
    corr = np.corrcoef(data[x], data[y])[0, 1]
    ax.text(
        0.02,
        0.9,
        f"r2={rvalue ** 2:.2f}, p={pvalue:.2g}, rmse={rmse:.2f}, corr={corr:.2f}",
        transform=ax.transAxes,
    )


def sanitize_metric_name(metric):
    return (
        metric.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("->", "to")
        .replace("(", "")
        .replace(")", "")
    )


def map_short(l):
    return ["/".join(ll.split("/")[-1:]) for ll in l]


MODELS_MAIN_EXPES = [
    "BAAI/bge-base-en-v1.5",
    "GritLM/GritLM-7B",
    # "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    "NousResearch/Llama-2-7b-hf",
    "Salesforce/SFR-Embedding-Mistral",
    "SmartComponents/bge-micro-v2",
    "TaylorAI/gte-tiny",
    "WhereIsAI/UAE-Large-V1",
    "avsolatorio/GIST-Embedding-v0",
    # "croissantllm/CroissantLLMBase",
    # "google/gemma-2b",
    "google/gemma-2b-it",
    # "google/gemma-7b",
    "google/gemma-7b-it",
    "infgrad/stella-base-en-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-small",
    "intfloat/multilingual-e5-small",
    "izhx/udever-bloom-560m",
    "jamesgpt1/sf_model_e5",
    "jspringer/echo-mistral-7b-instruct-lasttoken",
    "llmrails/ember-v1",
    "princeton-nlp/sup-simcse-bert-base-uncased",
    "sentence-transformers/LaBSE",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/allenai-specter",
    "sentence-transformers/average_word_embeddings_glove.6B.300d",
    "sentence-transformers/average_word_embeddings_komninos",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "sentence-transformers/gtr-t5-xl",
    "sentence-transformers/msmarco-bert-co-condensor",
    "sentence-transformers/sentence-t5-large",
    "sentence-transformers/sentence-t5-xl",
    "thenlper/gte-base",
    "thenlper/gte-large",
]
ALL_MODELS = [
    "sentence-transformers/average_word_embeddings_glove.6B.300d",
    "llmrails/ember-v1",
    "sentence-transformers/msmarco-bert-co-condensor",
    "google/gemma-7b",
    "sentence-transformers/sentence-t5-xl",
    "sentence-transformers/gtr-t5-large",
    "BAAI/bge-base-en-v1.5",
    "izhx/udever-bloom-560m",
    "NousResearch/Llama-2-7b-hf",
    "google/gemma-7b-it",
    "croissantllm/CroissantCool",
    "avsolatorio/GIST-Embedding-v0",
    "jamesgpt1/sf_model_e5",
    "Salesforce/SFR-Embedding-Mistral",
    "infgrad/stella-base-en-v2",
    "princeton-nlp/sup-simcse-bert-base-uncased",
    "sentence-transformers/average_word_embeddings_komninos",
    "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    "sentence-transformers/LaBSE",
    "SmartComponents/bge-micro-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-distilroberta-v1",
    "jspringer/echo-mistral-7b-instruct-lasttoken",
    "thenlper/gte-large",
    "intfloat/e5-small",
    "croissantllm/base_150k",
    "croissantllm/CroissantLLMBase",
    "thenlper/gte-base",
    "croissantllm/base_100k",
    "google/gemma-2b",
    "sentence-transformers/gtr-t5-base",
    "google/gemma-2b-it",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/gtr-t5-xl",
    "sentence-transformers/allenai-specter",
    "jinaai/jina-embedding-s-en-v1",
    "intfloat/multilingual-e5-small",
    "WhereIsAI/UAE-Large-V1",
    "croissantllm/base_50k",
    "sentence-transformers/sentence-t5-large",
    "GritLM/GritLM-7B",
    "croissantllm/base_5k",
    "intfloat/e5-large-v2",
    "TaylorAI/gte-tiny",
]


ALL_BUT_RANDOM = list(
    set(ALL_MODELS) - set(["HuggingFaceM4/tiny-random-LlamaForCausalLM"])
)



CROISSANT_CHECKPOINT_MODELS = [
    "croissantllm/base_5k",
    "croissantllm/base_50k",
    "croissantllm/base_100k",
    "croissantllm/base_150k",
    "croissantllm/CroissantLLMBase",
    "croissantllm/CroissantCool",
]

GEMMA_MODELS = [
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b",
    "google/gemma-7b-it",
]
TO_DISPLAY_GRAPH = [
    "croissantllm/CroissantCool",
    "croissantllm/CroissantLLMBase",

    "google/gemma-7b-it",
    "google/gemma-7b",
    "google/gemma-2b-it",
    "google/gemma-2b",

    "Salesforce/SFR-Embedding-Mistral",
    # UAE
    "WhereIsAI/UAE-Large-V1",
    # sf model
    "jamesgpt1/sf_model_e5",
    # average word
    "sentence-transformers/average_word_embeddings_glove.6B.300d",

    # allen ai
    "sentence-transformers/allenai-specter",

    # gtr
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "sentence-transformers/gtr-t5-xl",


    # Sentence transfo
    "sentence-transformers/LaBSE",

    # echo mistral
    "jspringer/echo-mistral-7b-instruct-lasttoken",

    # e5
    "intfloat/e5-small",
    "intfloat/e5-large-v2",

    # bloom
    "izhx/udever-bloom-560m",

    # llama
    "NousResearch/Llama-2-7b-hf",





    # stella

    "infgrad/stella-base-en-v2",
]
