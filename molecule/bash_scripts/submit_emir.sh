#!/bin/bash

# shellcheck disable=SC2054
models=("ContextPred" "GPT-GNN" "GraphMVP" "GROVER" "AttributeMask" "GraphLog" "GraphCL" "InfoGraph" "Not-trained" "MolBert" "ChemBertMLM-5M" "ChemBertMLM-10M" "ChemBertMLM-77M" "ChemBertMTR-5M" "ChemBertMTR-10M" "ChemBertMTR-77M" "ChemGPT-19M" "ChemGPT-4.7M" "ChemGPT-1.2B" "DenoisingPretrainingPQCMv4" "FRAD_QM9" "MolR_gat" "MolR_gcn" "MolR_tag" "MoleOOD_OGB_GIN" "MoleOOD_OGB_GCN" "MoleOOD_OGB_SAGE" "ThreeDInfomax")

for model in "${models[@]}"; do
    export MODELS=$model
    echo "Submitting job for model $MODELS"
    sbatch bash_scripts/emir_mol_MI_estimation.sh --export=MODELS
done
