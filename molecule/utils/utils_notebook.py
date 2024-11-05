import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm as tqdm
import numpy as np
import json
from sklearn.decomposition import PCA
import yaml
import networkx as nx
import matplotlib.patheffects as patheffects
from netgraph import Graph
from networkx.algorithms.community import louvain_communities
import datamol as dm
from scipy.cluster.hierarchy import linkage
from autorank import autorank

from utils import MolecularFeatureExtractor
from models.model_paths import get_model_path
from main import GROUPED_MODELS

LATEX_FIG_PATH = "../../emir-embedding-comparison/fig"

TASK_denomination = {
    "hERG": ["Tox", "Classification", 648],
    "hERG_Karim": ["Tox", "Classification", 13445],
    "AMES": ["Tox", "Classification", 7225],
    "DILI": ["Tox", "Classification", 475],
    "Carcinogens_Lagunin": ["Tox", "Classification", 278],
    "Skin__Reaction": ["Tox", "Classification", 404],
    "Tox21": ["Tox", "Classification", 7831],
    "ClinTox": ["Tox", "Classification", 1484],
    "LD50_Zhu": ["Tox", "Regression", 7385],
    "PAMPA_NCATS": ["Absorption", "Classification", 2035],
    "HIA_Hou": ["Absorption", "Classification", 578],
    "Pgp_Broccatelli": ["Absorption", "Classification", 1212],
    "Bioavailability_Ma": ["Absorption", "Regression", 640],
    "Caco2_Wang": ["Absorption", "Regression", 906],
    "CYP2C19_Veith": ["Metabolism", "Classification", 12665],
    "CYP2D6_Veith": ["Metabolism", "Classification", 13130],
    "CYP3A4_Veith": ["Metabolism", "Classification", 12328],
    "CYP1A2_Veith": ["Metabolism", "Classification", 12579],
    "CYP2C9_Veith": ["Metabolism", "Classification", 12092],
    "CYP2C9_Substrate_CarbonMangels": ["Metabolism", "Classification", 666],
    "CYP2D6_Substrate_CarbonMangels": ["Metabolism", "Classification", 664],
    "CYP3A4_Substrate_CarbonMangels": ["Metabolism", "Classification", 667],
    "BBB_Martins": ["Distribution", "Classification", 1975],
    "Lipophilicity_AstraZeneca": ["Distribution", "Regression", 4200],
    "Solubility_AqSolDB": ["Distribution", "Regression", 9982],
    "HydrationFreeEnergy_FreeSolv": ["Distribution", "Regression", 642],
    "PPBR_AZ": ["Distribution", "Regression", 1614],
    "VDss_Lombardo": ["Distribution", "Regression", 1130],
    "Half_Life_Obach": ["Excretion", "Regression", 667],
    "Clearance_Hepatocyte_AZ": ["Excretion", "Regression", 1020],
    "Clearance_Microsome_AZ": ["Excretion", "Regression", 1102],
}

MODELS = [
    "ContextPred",
    "GPT-GNN",
    "GraphMVP",
    "GROVER",
    # "EdgePred", # This model is especially bad and makes visualization hard
    "AttributeMask",
    "GraphLog",
    "GraphCL",
    "InfoGraph",
    "Not-trained",
    "MolBert",
    "ChemBertMLM-5M",
    "ChemBertMLM-10M",
    "ChemBertMLM-77M",
    "ChemBertMTR-5M",
    "ChemBertMTR-10M",
    "ChemBertMTR-77M",
    "ChemGPT-1.2B",
    "ChemGPT-19M",
    "ChemGPT-4.7M",
    "DenoisingPretrainingPQCMv4",
    "FRAD_QM9",
    "MolR_gat",
    "MolR_gcn",
    "MolR_tag",
    "MoleOOD_OGB_GIN",
    "MoleOOD_OGB_GCN",
    "MoleOOD_OGB_SAGE",
    "ThreeDInfomax",
]

MODELS_PATH = get_model_path(models=MODELS)


def plot_embeddings(
    DATASET,
    MODELS=MODELS,
    n_cols=4,
    figsize=5,
    alpha=0.5,
    n_mols=10000,
    desc="mw",
    min_hue=0,
    max_hue=1,
):
    with open(f"data/{DATASET}/smiles.json", "r") as f:
        smiles = json.load(f)

    random_idx = np.random.choice(len(smiles), n_mols, replace=False)
    smiles_cons = [smiles[i] for i in random_idx]
    mols = [dm.to_mol(s) for s in smiles_cons]
    df_desc = dm.descriptors.batch_compute_many_descriptors(mols)
    hue = (df_desc[desc] - df_desc[desc].min()) / (
        df_desc[desc].max() - df_desc[desc].min()
    )

    plt.hist(hue, bins=100)
    plt.show()

    hue = hue.clip(min_hue, max_hue)
    fig, axes = plt.subplots(
        n_cols, len(MODELS) // n_cols, figsize=(figsize * n_cols / 2, figsize)
    )
    axes = axes.flatten()

    for i, model in enumerate(MODELS):
        embeddings = np.load(f"data/{DATASET}/{model}.npy", mmap_mode="r")
        embeddings = embeddings[random_idx]
        # nromalize embeddings
        embeddings = (embeddings - embeddings.mean(axis=0)) / (
            embeddings.std(axis=0) + 1e-8
        )

        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        df = pd.DataFrame(embeddings_pca, columns=[f"PC{i}" for i in range(1, 3)])
        df["smiles"] = smiles_cons

        sns.scatterplot(
            data=df,
            x="PC1",
            y="PC2",
            ax=axes[i],
            hue=hue,
            cmap="viridis",
            legend=False,
            alpha=alpha,
        )
        axes[i].set_title(model)
        axes[i].set_xlabel("PC1")
        axes[i].set_ylabel("PC2")
        del embeddings
    plt.tight_layout()
    plt.show()


def get_MI_df(
    DATASET,
    results_dir_list,
    args_to_add=[
        "cond_modes",
        "marg_modes",
        "ff_layers",
        "ff_hidden_dim",
        "batch_size",
    ],
):
    full_df_MI = []

    for results_dir in tqdm(results_dir_list):
        RESULTS_PATH = f"results/{DATASET}/{results_dir}"

        with open(RESULTS_PATH + "/args.yaml", "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)

        all_df = []
        for file in os.listdir(RESULTS_PATH):
            if file.endswith(".csv"):
                all_df.append(pd.read_csv(os.path.join(RESULTS_PATH, file)))
        all_df = pd.concat(all_df)
        for arg in args_to_add:
            all_df[arg] = args[arg]
        full_df_MI.append(all_df)

    df = pd.concat(full_df_MI)

    del df["I(X|Y)"]
    del df["I(Y->X)"]

    df_copy = df.copy()

    df_copy["I(Y->X)"] = df["I(X->Y)"]
    df_copy["Y"] = df["X"]
    df_copy["X"] = df["Y"]

    df_copy.set_index(["X", "Y"] + args_to_add, inplace=True)

    df = df.join(df_copy["I(Y->X)"], on=["X", "Y"] + args_to_add)
    df["I(X->Y)/dim"] = df["I(X->Y)"] / df["Y_dim"]
    df["I(Y->X)/dim"] = df["I(Y->X)"] / df["X_dim"]

    return df.fillna(0)


def plot_cmap(
    df,
    keys,
    cmap="copper",
    vmin=None,
    vmax=None,
    center=None,
    values=False,
    same_linkage=True,
    title="",
    linkage_method="ward",
):
    if vmax is None:
        vmax = [None] * len(keys)
    if vmin is None:
        vmin = [None] * len(keys)
    for i, key in enumerate(keys):
        df_pivot = df.pivot_table(index="X", columns="Y", values=key, aggfunc="mean")
        if same_linkage:
            link = linkage(df_pivot, method=linkage_method)
        else:
            link = None
        cluster = sns.clustermap(
            df_pivot,
            row_linkage=link,
            col_linkage=link,
            cmap=cmap,
            figsize=(8, 8),
            vmin=vmin[i],
            vmax=vmax[i],
            center=center,
            annot=values,
            dendrogram_ratio=(0.15, 0.15),
        )
        if len(keys) == 1:
            return
        cluster.savefig("fig/cluster_{}.png".format(i))
        plt.clf()

    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, len(keys), figsize=(8 * len(keys), 8))

    for i, key in enumerate(keys):
        if len(keys) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(mpimg.imread("fig/cluster_{}.png".format(i)))
        ax.axis("off")
        ax.set_title(key)

    fig.suptitle(title)


def plot_com(
    df_in,
    weight_col="I(X->Y)/dim",
    cmap="coolwarm",
    edge_cmap="coolwarm",
    node_cmap_offset=0,
    edge_cmap_offset=0,
    min_alpha=0.0,
    max_alpha=1.0,
    min_edge_width=0.0,
    max_edge_width=1,
    edge_pow=2,
    com_resolution=1.1,
    figsize=10,
    clip_min_values_alpha=0,
    com_pad_by=0.001,
    fontsize=15,
    sparsity=1,
    node_size=100,
    scale=(1, 2),
    communities=None,
    nodes_to_display=None,
):
    if nodes_to_display is None:
        nodes_to_display = df_in.X.unique()
    df = df_in[df_in.X != df_in.Y].copy()
    df[weight_col] = (df[weight_col] - df[weight_col].min()) / (
        df[weight_col].max() - df[weight_col].min()
    )

    table = df.pivot_table(index="X", columns="Y", values=weight_col, aggfunc="mean")
    G = nx.from_pandas_adjacency(table, create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))

    if sparsity < 1:
        df[weight_col + "_mask"] = 1
        df[weight_col + "_mask"] *= df[weight_col] > df.groupby("X")[
            weight_col
        ].transform(lambda x: np.percentile(x.unique(), sparsity * 100))

        table_un = df.pivot_table(
            index="X", columns="Y", values=weight_col + "_mask", aggfunc="mean"
        )
        G_un = nx.from_pandas_adjacency(table_un, create_using=nx.DiGraph)

        G_un.remove_edges_from(nx.selfloop_edges(G_un))
        G_un.remove_edges_from(
            [(u, v) for u, v, d in G_un.edges(data=True) if d["weight"] == 0]
        )

    if communities is None:
        communities = louvain_communities(G, resolution=com_resolution)
        communities = list(communities)
    cmap = sns.color_palette(
        cmap,
        as_cmap=True,
    )
    edge_cmap = sns.color_palette(edge_cmap, as_cmap=True)

    all_avg_weights = np.array(
        [
            np.median([d[2]["weight"] for d in G.out_edges(n, data=True)])
            for n in G.nodes()
        ]
    )
    all_avg_incomes = np.array(
        [
            np.median([d[2]["weight"] for d in G.in_edges(n, data=True)])
            for n in G.nodes()
        ]
    )

    avg_weight = {
        n: (
            np.median([d[2]["weight"] for d in G.out_edges(n, data=True)])
            - all_avg_weights.min()
        )
        / (all_avg_weights.max() - all_avg_weights.min())
        for n in G.nodes()
    }
    avg_income = {
        n: (
            np.median([d[2]["weight"] for d in G.in_edges(n, data=True)])
            - all_avg_incomes.min()
        )
        / (all_avg_incomes.max() - all_avg_incomes.min())
        for n in G.nodes()
    }

    node_to_community = {
        node: i for i, community in enumerate(communities) for node in community
    }

    node_color = {node: cmap(avg_weight[node] + node_cmap_offset) for node in G.nodes()}
    node_edge_color = {
        node: cmap(avg_income[node] + node_cmap_offset) for node in G.nodes()
    }
    node_labels = {node: node for node in G.nodes()}

    edge_color = {
        edge: edge_cmap(G.edges[edge]["weight"] + edge_cmap_offset)
        for edge in G.edges()
    }

    # normalize edge alpha
    if sparsity < 1:
        edge_alpha = {
            edge: G.edges[edge]["weight"] if edge in G_un.edges() else 0
            for edge in G.edges()
        }
    else:
        edge_alpha = {edge: G.edges[edge]["weight"] for edge in G.edges()}

    min_edge = min(edge_alpha.values())
    edge_alpha = {
        edge: ((edge_alpha[edge] - min_edge) / (max(edge_alpha.values()) - min_edge))
        * (max_alpha - min_alpha)
        + min_alpha
        for edge in edge_alpha
    }

    # edge width

    edge_width = {edge: G.edges[edge]["weight"] for edge in G.edges()}
    min_edge = min(list(edge_width.values()))
    edge_width = {
        edge: (edge_width[edge] - min_edge)
        / (max(edge_width.values()) - min_edge) ** edge_pow
        * (max_edge_width - min_edge_width)
        + min_edge_width
        for edge in edge_width
    }

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    node_layout_kwargs = dict(node_to_community=node_to_community, pad_by=com_pad_by)

    node_to_label = {
        node: node if node in nodes_to_display else "" for node in G.nodes()
    }

    graph = Graph(
        G,
        node_layout_kwargs=node_layout_kwargs,
        node_layout="community",
        node_color=node_color,
        node_labels=node_to_label,
        edge_color=edge_color,
        ax=ax,
        node_label_fontdict={"fontsize": fontsize, "fontweight": "bold"},
        node_edge_color=node_edge_color,
        edge_layout="straight",
        edge_alpha=edge_alpha,
        arrows=True,
        prettify=True,
        scale=scale,
        edge_width=edge_width,
        node_size=node_size,
    )

    # add white contour to all texts in the figure
    for text in plt.gca().texts:
        text.set_path_effects(
            [patheffects.Stroke(linewidth=4, foreground="white"), patheffects.Normal()]
        )

    plt.savefig(
        f"{LATEX_FIG_PATH}/molecule/MI_graph_v2_{sparsity}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


def get_ranked_df(
    df,
    path="results/TDC_ADMET_SCAFF.csv",
    split_on=None,
    COLUMS_SPLIT="cond_modes",
    n_runs=10,
    information="I(X->Y)/dim",
):
    df_downs = pd.read_csv(path)
    n_models = df_downs.embedder.nunique()
    df_downs["run_id"] = 0

    for embedder in df_downs.embedder.unique():
        for dataset in df_downs.dataset.unique():
            n_tot_eval = df_downs[
                (df_downs.embedder == embedder) & (df_downs.dataset == dataset)
            ].shape[0]
            n_expe_per_run = n_tot_eval // n_runs
            df_downs.loc[
                (df_downs.embedder == embedder) & (df_downs.dataset == dataset),
                "run_id",
            ] = np.arange(
                0,
                df_downs[
                    (df_downs.embedder == embedder) & (df_downs.dataset == dataset)
                ].shape[0],
            )
            df_downs.loc[
                (df_downs.embedder == embedder) & (df_downs.dataset == dataset),
                "run_id",
            ] = df_downs.loc[
                (df_downs.embedder == embedder) & (df_downs.dataset == dataset),
                "run_id",
            ].map(
                lambda x: x // n_expe_per_run
            )

    df_avg = df[df.X != df.Y].groupby(["X", COLUMS_SPLIT]).median()
    df_avg["information"] = df_avg[information]
    df_avg.rename(columns={"X": "embedder"}, inplace=True)

    df_downs = df_downs.join(
        df_avg.reset_index().set_index("X")[["information", COLUMS_SPLIT]],
        on="embedder",
        how="inner",
    )

    df_downs["task_category"] = df_downs.dataset.map(lambda x: TASK_denomination[x][0])
    df_downs["task_type"] = df_downs.dataset.map(lambda x: TASK_denomination[x][1])
    df_downs["task_size"] = df_downs.dataset.map(lambda x: TASK_denomination[x][2])
    df_downs["task_size_bin"] = df_downs.task_size.map(
        lambda x: "small" if x < 700 else "medium" if x < 7000 else "large"
    )
    df_downs["meanrank_information"] = np.nan
    df_downs["meanrank_metric"] = np.nan

    if split_on is None:
        df_ranked = []
        for dataset in df_downs.dataset.unique():
            df_to_rank = df_downs[df_downs.dataset == dataset].pivot_table(
                index="run_id",
                columns=["embedder"],
                values="metric_test",
                aggfunc="mean",
            )
            res = autorank(
                df_to_rank, alpha=0.05, verbose=False, force_mode="nonparametric"
            ).rankdf
            res.rename(columns={"meanrank": "meanrank_metric"}, inplace=True)
            for model in df_downs.embedder.unique():
                df_downs.loc[
                    (df_downs.dataset == dataset) & (df_downs.embedder == model),
                    "meanrank_metric",
                ] = res.loc[model, "meanrank_metric"]

        # add global rank
        df_to_rank = df_downs.pivot_table(
            index=["dataset", "run_id"],
            columns=["embedder"],
            values="metric_test",
            aggfunc="mean",
        )
        res = autorank(
            df_to_rank, alpha=0.05, verbose=False, force_mode="nonparametric"
        ).rankdf
        res.rename(columns={"meanrank": "global_meanrank_metric"}, inplace=True)
        df_downs = df_downs.join(res["global_meanrank_metric"], on=["embedder"])

    else:
        for value in df_downs[split_on].unique():
            for dataset in df_downs.dataset.unique():
                df_to_rank = df_downs[
                    (df_downs[split_on] == value) & (df_downs.dataset == dataset)
                ].pivot_table(
                    index="run_id",
                    columns=["embedder"],
                    values="metric_test",
                    aggfunc="mean",
                )
                if df_to_rank.shape[0] == 0:
                    continue
                res = autorank(
                    df_to_rank, alpha=0.05, verbose=False, force_mode="nonparametric"
                ).rankdf
                res.rename(columns={"meanrank": "meanrank_metric"}, inplace=True)
                for model in df_downs.embedder.unique():
                    df_downs.loc[
                        (df_downs[split_on] == value)
                        & (df_downs.dataset == dataset)
                        & (df_downs.embedder == model),
                        "meanrank_metric",
                    ] = res.loc[model, "meanrank_metric"]
            # add global rank
            df_to_rank = df_downs[df_downs[split_on] == value].pivot_table(
                index=["dataset", "run_id"],
                columns=["embedder"],
                values="metric_test",
                aggfunc="mean",
            )
            res = autorank(
                df_to_rank, alpha=0.05, verbose=False, force_mode="nonparametric"
            ).rankdf
            res.rename(columns={"meanrank": "global_meanrank_metric"}, inplace=True)
            df_downs.loc[df_downs[split_on] == value, "global_meanrank_metric"] = (
                df_downs[df_downs[split_on] == value].embedder.map(
                    res["global_meanrank_metric"]
                )
            )

    for x in df_downs[COLUMS_SPLIT].unique():
        df_to_rank = df_downs[df_downs[COLUMS_SPLIT] == x].pivot_table(
            index="dataset", columns="embedder", values="information", aggfunc="mean"
        )
        res = autorank(
            df_to_rank, alpha=0.05, verbose=False, force_mode="nonparametric"
        ).rankdf
        res.rename(columns={"meanrank": "meanrank_information"}, inplace=True)
        df_downs.loc[df_downs[COLUMS_SPLIT] == x, "meanrank_information"] = df_downs[
            df_downs[COLUMS_SPLIT] == x
        ].embedder.map(res["meanrank_information"])
    return df_downs


def get_DTI_rank_df(
    df,
    dataset="KIBA",
    metric="clustering_1",
    split_on=None,
    COLUMS_SPLIT="cond_modes",
    order="descending",
):
    path = f"results/{dataset}_DTI.csv"
    df_downs = pd.read_csv(path)

    df_avg = df[df.X != df.Y].groupby(["X", COLUMS_SPLIT]).median()
    df_avg["information"] = df_avg["I(X->Y)/dim"]
    df_avg.rename(columns={"X": "embedder"}, inplace=True)

    df_downs = df_downs.join(
        df_avg.reset_index().set_index("X")[["information", COLUMS_SPLIT]],
        on="embedder",
        how="inner",
    )

    if split_on is None:
        df_to_rank = df_downs.pivot_table(
            index="target", columns=["embedder"], values=metric, aggfunc="mean"
        )
        df_to_rank.columns.name = None
        df_to_rank.index.name = None

        res = autorank(
            df_to_rank,
            alpha=0.05,
            verbose=False,
            order=order,
            # force_mode="nonparametric",
        ).rankdf
        res.rename(columns={"meanrank": "meanrank_metric"}, inplace=True)

        df_downs = df_downs.join(res[["meanrank_metric"]], on=["embedder"])
    else:
        for value in df_downs[split_on].unique():
            df_to_rank = df_downs[df_downs[split_on] == value].pivot_table(
                index="target", columns=["embedder"], values=metric, aggfunc="mean"
            )
            res = autorank(
                df_to_rank,
                alpha=0.05,
                verbose=False,
                order=order,  # force_mode="nonparametric"
            ).rankdf
            res.rename(columns={"meanrank": "meanrank_metric"}, inplace=True)
            df_downs.loc[df_downs[split_on] == value, "meanrank_metric"] = df_downs[
                df_downs[split_on] == value
            ].embedder.map(res["meanrank_metric"])

    df_downs["meanrank_information"] = np.nan
    for x in df_downs[COLUMS_SPLIT].unique():
        df_to_rank = df_downs[df_downs[COLUMS_SPLIT] == x].pivot_table(
            index="target", columns="embedder", values="information", aggfunc="mean"
        )
        res = autorank(
            df_to_rank,
            alpha=0.05,
            verbose=False,
            # force_mode="nonparametric"
        ).rankdf
        res.rename(columns={"meanrank": "meanrank_information"}, inplace=True)
        df_downs.loc[df_downs[COLUMS_SPLIT] == x, "meanrank_information"] = df_downs[
            df_downs[COLUMS_SPLIT] == x
        ].embedder.map(res["meanrank_information"])
    return df_downs


def process_dataset_name(dataset):
    return (
        dataset.replace("CarbonMangels", "Carb.")
        .replace("Substrate", "Sub.")
        .replace("_AstraZeneca", "")
        .replace("_AZ", "")
        .replace("HydrationFreeEnergy_", "")
        .replace("__", " ")
        .replace("_", " ")
        .replace("Clearance", "Clear.")
        .replace("NCATS", "")
        .replace("Lagunin", "")
        .replace("Broccatelli", "")
        .replace("Ma", "")
        .replace("Hou", "")
    )


def prerpocess_emb_name(x):
    return (
        x.replace("DenoisingPretrainingPQCMv4", "3D-denoising")
        .replace("Chem", "")
        .replace("ThreeDInfomax", "3D-Infomax")
        .replace("_OGB", "")
    )
