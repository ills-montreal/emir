#!/bin/bash
#SBATCH --job-name=mds-fingerprint-continuous
#SBATCH --account=def-ibenayed
#SBATCH --time=0-24:00:00
#SBATCH --mem=2008G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir

cp -r /home/fransou/scratch/emir/emir .
cd emir/molecule
cp -r /home/fransou/scratch/DATA/EMIR/data/ZINC data

module load python/3.10
module load scipy-stack
source /home/fransou/EMIR/bin/activate

python make_fingerprint_continuous.py --dataset ZINC --fp-length 1024 --out-dim 64 --out-dir /home/fransou/scratch/DATA/EMIR/data/fingerprint_continuous
