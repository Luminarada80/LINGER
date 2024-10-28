#!/bin/bash -l

#SBATCH --job-name=linger_cell_type_grn_inference
#SBATCH --output=Linger_Cell_Type_Grn_Inference_Results.txt
#SBATCH --error=Linger_Cell_Type_Grn_Inference_Errors.err
#SBATCH -p compute
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=50G

source /gpfs/Home/esm5360/.bashrc

conda activate LINGER_1.92

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MESC_PIPELINE/

python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MESC_PIPELINE/Step_050.Create_Cell_Type_GRN.py