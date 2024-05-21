

# Text Embedding Evaluation using Information Sufficiency



## Folder Structure

```
├── analysis
│   ├── 2D Clustering Projection.ipynb
│   ├── Analysis MI specific DS.ipynb
│   ├── Check BIC.ipynb
│   ├── Classification results.ipynb
│   ├── Correlation with downstream tasks performance.ipynb
│   ├── Croissant LLM Training.ipynb
│   ├── Cross MIS from metadata.ipynb
│   ├── Cross MIS.ipynb
│   ├── Datasets statistics.ipynb
│   ├── Detailed MTEB.ipynb
│   ├── exported_data # Contains the raw data used in the analysis
│   │   ├── classification_many_3_avg.csv
│   │   ├── dataset_stats.csv
│   │   ├── df_mteb_avg.csv
│   │   ├── normalized_13.df
│   │   ├── normalized_m10.df
│   │   ├── normalized_m12.df
│   │   ├── normalized_m16.df
│   │   ├── normalized_m1.df
│   │   ├── normalized_m2.df
│   │   ├── normalized_m4.df
│   │   └── normalized_multi_ds.df
│   ├── Gemma Instruct Finetuning.ipynb
│   ├── Impact of number of models.ipynb
│   ├── Models summary.ipynb
│   ├── Modes Analysis.ipynb
│   ├── Modes Impact.ipynb
│   ├── modes_kmeans.csv
│   ├── modes_stats.csv
│   ├── MTEB and Classiffications overal.ipynb
│   ├── MTEB Correlations.ipynb
│   ├── MTEB Correlations No size Normalization.ipynb
│   ├── Mutual information graph.ipynb
│   ├── Overal table.ipynb
│   ├── Size vs Informativeness.ipynb
│   └── visu_utils.py
├── Readme.md
├── scripts
│   ├── cache_models.py # Cache the huggingface models, useful on a cluster
│   ├── emb_datasets.py # Loaders for the datasets
│   ├── evaluate_mis.py # Evaluate the information sufficiency/MI between two datasets
│   ├── evaluate_mis_target.py # batch evaluation for a given target dataset
│   ├── generate_embeddings.py # generate the embeddings and save them
│   ├── merge_df.py # Merge results for analysis
│   ├── pack_classification_results.py # Pack the classification results
│   ├── train_eval_cross_embeddings_prediction.py # Train and evaluate the cross embeddings prediction
│   └── train_eval_embedding_for_classification.py # Train and evaluate the embeddings for classification
└── slurms
    ├── compute_classification_perfs_concat.sh
    ├── compute_classification_perfs.sh
    ├── compute_cross_embeddings_prediction.sh
    ├── compute_mis_target.sh
    ├── generate_embeddings_classification_tasks.sh
    └── generate_embeddings.sh
```

