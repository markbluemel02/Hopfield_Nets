#!/bin/bash

#SBATCH --partition=cidbn
# SBATCH --partition=scc-cpu
#SBATCH --job-name=joblib-optuna  
#SBATCH --ntasks=1                                            
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=./logfiles/output-mult-%j.out
# ##SBATCH --output=output-%x.%j.log

export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

source ~/.bashrc
module load miniforge3
source activate infomorphic_env
#conda activate infomorphic_env

# export ALTEXPDIR=/user/mblueme/u25725/.project/dir.project/mark/revision/data/Hopfield
# export ALTEXPDIR=/user/mblueme/u26551/.project/dir.project/mark/data/Hopfield
python src/simulations.py --neurons 800 --reps 1
