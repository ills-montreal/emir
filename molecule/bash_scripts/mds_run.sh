#!/bin/bash
#SBATCH --job-name=mds-fingerprint-continuous
#SBATCH --account=def-ibenayed
#SBATCH --time=0-24:00:00
#SBATCH --mem=2008G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out

export DATASET="ClinTox"
export FP_LENGTH=1024
export OUT_DIM=64

echo "Running script"
cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir

echo "Copying files"
cp -r /home/fransou/scratch/emir .
cd emir/molecule
cp -r /home/fransou/scratch/DATA/EMIR/data/$DATASET data

echo "Loading modules"
module load python/3.10
module load scipy-stack
source /home/fransou/EMIR/bin/activate

echo "Running script"
python make_fingerprint_continuous.py --dataset $DATASET --fp-length $FP_LENGTH --out-dim $OUT_DIM --descriptors cats --out-dir /home/fransou/scratch/DATA/EMIR/data/fingerprint_continuous
