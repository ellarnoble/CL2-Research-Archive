#!/bin/bash --login
#SBATCH -J coursework
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 4-00:00:00
#SBATCH --mem=32G
#SBATCH --error=coursework_%j.err

module purge
module load python/3.13.1

source env/bin/activate

python ./script.py
