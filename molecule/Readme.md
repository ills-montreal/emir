# :pill: Usage of EMIR for molecular data

This directory contains the code to use EMIR for molecular data. The code is written in Python and uses the RDKit/Datamol library\cite{} to handle molecular data.
The datasets are imported from the Therapeutic Data Commons (TDC) plateform\cite{}.

## :clipboard: Installation

To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt # TOCHANGE

cd external_repo/pre-training-via-denoising
pip install -e .
```

## :file_folder: Data Preprocessing
Various descriptors use a 3-dimensional representation of the molecules. 
To generate these representations, we use the [precompute_3D.py](precompute_3D.py) script.
```bash
python precompute_3D.py --dataset <dataset_name>
```

This command will generate a file named "<dataset_name>_3d.sdf" containing the 3D representations of the molecules in the dataset.
The molecular descriptors can then be computed using the script [precompute_molf_descriptors.py](precompute_molf_descriptors.py), which uses descriptors from the molfeat\cite{} library.
```bash
python precompute_molf_descriptors.py --dataset <dataset_name> --descriptors <[optional] descriptor_names>
```
Using this script will also preprocess the datasets, removing molecules that cannot be turned into 2D representations for our models.

## :chart_with_upwards_trend: EMIR

To run EMIR, you can use the following command:
```bash
python main.py --dataset <dataset_name> --X <models_names> --Y <models_names> --out-dir <output_dir>
```

This command will train the models specified in the `--X` and `--Y` arguments on the dataset `<dataset_name>`, and save the results in the directory `<output_dir>`.
Once the Mutual information is computed, the results can be visualized using the notebooks

## :microbe: Downstream tasks

The various models considered can be trained on downstream tasks using the script [main_downstream.py](main_downstream.py).
```bash
python main_downstream.py --dataset <dataset_name> --embedders <models_names> --n-runs <number_of_runs> --split-method <random/scaffold> --hidden-dim <hidden_dim> --n-layers <n_layers> --n-epochs <n_epochs> --d-rate <dropout_rate> --lr <learning_rate> --batch-size <batch_size>
```

This command will train the models on the single instance tasks using the dataset `<dataset_name>`, and save the results in the directory `<output_dir>`.

For DTI tasks, the script [main_downstream_dti.py](main_downstream_dti.py) can be used.
```bash
python main_downstream.py --dataset <dataset_name> --embedders <models_names>
```