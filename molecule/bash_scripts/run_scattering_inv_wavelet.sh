#!/bin/bash
#SBATCH --job-name=precompute-3d
#SBATCH --account=def-ibenayed
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out

cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir

cp -r /home/fransou/scratch/emir/emir .
cd emir/molecule

module load python/3.10
module load scipy-stack

source /home/fransou/EMIR/bin/activate
cp -r /home/fransou/scratch/DATA/EMIR/data/ClinTox data

python utils/scattering_wavelet.py --dataset ClinTox --i-start $start_i --out-dir /home/fransou/scratch/DATA/EMIR/data/scattering_wavelet
