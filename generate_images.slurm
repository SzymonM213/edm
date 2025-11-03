#!/bin/bash
#SBATCH --job-name=img-gen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Launch training
singularity exec --nv --bind ~/:/scratch edm.sif torchrun --standalone --nproc_per_node=2 generate-vel.py --outdir=$outdir --seeds=0-49999 --subdirs \
        --network=$model_path --steps=512 --eta=pokar