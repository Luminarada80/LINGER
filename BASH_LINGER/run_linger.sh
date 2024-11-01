#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem-per-cpu=16G

source /gpfs/Home/esm5360/.bashrc
conda activate LINGER_1.92
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MESC_PIPELINE/

LOG_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER"

# Dynamically set the job name
scontrol update JobID=$SLURM_JOB_ID JobName=linger_${SAMPLE_NUM}

# Set output and error files dynamically
exec > "${LOG_DIR}/Linger_Results_${SAMPLE_NUM}.txt" 2> "${LOG_DIR}/Linger_Errors_${SAMPLE_NUM}.err"

GENOME='mm10'
METHOD='scNN'
CELLTYPE='all'
ACTIVEF='ReLU'

UZUN_LAB_DIR='/gpfs/Labs/Uzun'
DATA_DIR="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
TSS_MOTIF_INFO_PATH="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"
OUTPUT_DIR="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_TRAINED_MODEL"
RESULTS_DIR="${UZUN_LAB_DIR}/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS"

# Define paths for RNA and ATAC data
RNA_DATA_PATH="${DATA_DIR}/subsampled_RNA_${SAMPLE_NUM}.csv"
ATAC_DATA_PATH="${DATA_DIR}/subsampled_ATAC_${SAMPLE_NUM}.csv"
SAMPLE_DATA_DIR="${OUTPUT_DIR}/sample_${SAMPLE_NUM}"
SAMPLE_OUTPUT_DIR="${OUTPUT_DIR}/sample_${SAMPLE_NUM}/"
SAMPLE_RESULTS_DIR="${RESULTS_DIR}/sample_${SAMPLE_NUM}"

# Create directories for output
mkdir -p "${SAMPLE_DATA_DIR}" "${SAMPLE_OUTPUT_DIR}" "${SAMPLE_RESULTS_DIR}"

# Run each step of the pipeline
echo "Processing sample number ${SAMPLE_NUM}..."

# echo running Step_010.Linger_Load_Data.py
# python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_010.Linger_Load_Data.py \
#   --rna_data_path "$RNA_DATA_PATH" \
#   --atac_data_path "$ATAC_DATA_PATH" \
#   --data_dir "$DATA_DIR" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR"

# echo running Step_020.Linger_Training.py
# python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_020.Linger_Training.py \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --genome "$GENOME" \
#   --method "$METHOD" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR" \
#   --activef "$ACTIVEF"

# echo running Step_030.Create_Cell_Population_GRN.py
# python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_030.Create_Cell_Population_GRN.py \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --genome "$GENOME" \
#   --method "$METHOD" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR" \
#   --activef "$ACTIVEF"

# echo running Step_040.Homer_Motif_Finding.py
# python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_040.Homer_Motif_Finding.py \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --genome "$GENOME" \
#   --output_dir "$SAMPLE_OUTPUT_DIR"

# echo running Step_050.Create_Cell_Type_GRN.py
# python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_050.Create_Cell_Type_GRN.py \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --genome "$GENOME" \
#   --method "$METHOD" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR" \
#   --celltype "$CELLTYPE"

echo running Step_060.Analyze_Results.py
python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_060.Analyze_Results.py \
  --cell_type "mESC" \
  --sample_num "$SAMPLE_NUM"