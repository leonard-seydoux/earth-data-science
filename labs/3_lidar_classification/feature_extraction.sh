#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x.%A.%a.log
#SBATCH --partition=ncpushort,ncpum,ncpu,ncpulong

# Header
echo Working directory: `pwd`
echo Hostname: `hostname`
echo Date: `date`

# Environment
module purge
eval "$(conda shell.bash hook)"

# Anaconda environment
conda activate nuts_lyon
echo Python: `which python`

python feature_extraction.py