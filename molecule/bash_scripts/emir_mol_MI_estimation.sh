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

export LENGTH=2048
export DATASET=HIV

cd $SLURM_TMPDIR
mkdir tmp_dir
cd tmp_dir

cp -r /home/fransou/scratch/emir .

module load python/3.10
module load scipy-stack
source /home/fransou/EMIR/bin/activate

cd emir
pip install -e .

cd molecule
mkdir data/$DATASET
cp -r /home/fransou/scratch/DATA/EMIR/data/$DATASET/**$LENGTH** data/$DATASET/.
cp -r /home/fransou/scratch/DATA/EMIR/backbone_pretrained_models ./

export WANDB_MODE=offline


echo "Running script"
python main.py --dataset $DATASET --fp-length $LENGTH
