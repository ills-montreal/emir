#!/bin/bash
#SBATCH --job-name=emir_molecule
#SBATCH --account=def-ibenayed
#SBATCH --time=0-18:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out

export LENGTH=512
export DATASET=HIV

cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir

mkdir emir
cd emir
cp /home/fransou/scratch/emir/pyproject.toml .
cp /home/fransou/scratch/emir/Readme.md .
cp -r /home/fransou/scratch/emir/emir .
mkdir molecule
cd molecule
cp /home/fransou/scratch/emir/molecule/**.py .
cp -r /home/fransou/scratch/emir/molecule/utils .
cp -r /home/fransou/scratch/emir/molecule/models .
cp -r /home/fransou/scratch/emir/molecule/backbone_pretrained_models .
mkdir data
mkdir data/$DATASET
cp -r /home/fransou/scratch/emir/molecule/data/$DATASET/**$LENGTH** data/$DATASET/.
cd ..


module load python/3.10
source /home/fransou/scratch/env/bin/activate
pip install -e .
cd molecule

export WANDB_MODE=offline


echo "Running script"
python main.py --dataset HIV --fp-length 256 --n-runs 3
cp -r data /home/fransou/scratch/emir/molecule/data
cp -r wandb /home/fransou/scratch/emir/molecule/wandb
