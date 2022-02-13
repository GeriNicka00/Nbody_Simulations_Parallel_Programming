#!/bin/bash
# ==================
# Nbody_MPI.sh
# ==================

#SBATCH --job-name=Nbody_MPI
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --time=0:1:0
#SBATCH --mem-per-cpu=100M

module add languages/anaconda3/2020-3.8.5

cd $SLURM_SUBMIT_DIR


mpiexec -n 28 python Nbody_MPI.py 500 200
