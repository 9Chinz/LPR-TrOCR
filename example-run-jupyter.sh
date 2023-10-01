#!/bin/bash
#SBATCH -A <Your name>
#SBATCH --gres=gpu:1
#SBATCH --job-name=<job name>
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -o out_%j.txt
#SBATCH -e err_%j.txt

source activate tf
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

which python

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=6789