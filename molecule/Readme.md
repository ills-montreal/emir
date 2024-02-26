# :pill: Usage of EMIR for molecular data

This directory contains the code to use EMIR for molecular data. The code is written in Python and uses the RDKit/Datamol library\cite{} to handle molecular data.
The datasets are imported from the Therapeutic Data Commons (TDC) plateform\cite{}.

## :clipboard: Installation

To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt # TOCHANGE
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

Finally, wavelet features can be computed using the [utils/precompute_wavelet.py](utils/precompute_wavelet.py) script (from the molecule folder).
```bash
python precompute_wavelet.py --dataset <dataset_name> --i0 <i0> --i1 <i1>
```

Where i0 and i1 are the indices of the first and last molecule to process, used for parallelization.

## :chart_with_upwards_trend: EMIR

