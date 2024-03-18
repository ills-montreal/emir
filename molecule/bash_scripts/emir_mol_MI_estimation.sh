#!/bin/bash
#SBATCH --job-name=emir_molecule
#SBATCH --account=def-ibenayed
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out

export DATASET=ZINC

echo "Starting job on dataset $DATASET  and model $MODELS"

cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir

cp -r /home/fransou/scratch/emir .

module load python/3.10
module load scipy-stack
source /home/fransou/EMIR/bin/activate

cd emir/molecule
cp -r /home/fransou/scratch/DATA/EMIR/data/$DATASET data
cp -r /home/fransou/scratch/DATA/EMIR/backbone_pretrained_models ./

export WANDB_MODE=offline

echo "Running script"
python main.py --dataset $DATASET --models $MODELS --out-dir /home/fransou/scratch/DATA/results --batch-size 512 --n-epochs 20 --n-epochs-marg 20 --ff-layers 1 --fp-length 0
cp -r wandb /home/fransou/scratch/DATA/results/wandb