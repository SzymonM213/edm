#!/bin/bash
#SBATCH --job-name=bitcoin-farm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Launch training
singularity exec --nv --bind ~/:/scratch edm.sif torchrun --standalone --nproc_per_node=4 train.py --outdir=training-runs \
      --data=datasets/cifar-10-32x32.zip --cond=1 --batch=200 --duration=200 --precond=uvel