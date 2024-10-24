#!/bin/bash -l

#SBATCH --job-name=linger_training
#SBATCH --output=Linger_Training_Results.txt
#SBATCH --error=Linger_Training_Errors.err
#SBATCH -p compute
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=32G

source /gpfs/Home/esm5360/.bashrc

conda activate LINGER

python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/Step_035.Linger_Training.py