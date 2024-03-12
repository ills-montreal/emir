#!/bin/bash
#SBATCH --job-name=precompute-3d
#SBATCH --account=def-ibenayed
#SBATCH --time=0-06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir


cp -r /home/fransou/scratch/emir .
cd emir/molecule

module load python/3.10
source /home/fransou/scratch/env/bin/activate
export WANDB_MODE=offline


echo "Running training script"
python precompute_3d.py
cp -r data /home/fransou/scratch/emir/molecule/data

