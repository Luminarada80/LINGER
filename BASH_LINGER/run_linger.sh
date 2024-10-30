#!/bin/bash -l

#SBATCH --job-name=linger
#SBATCH --output=Linger_Results.txt
#SBATCH --error=Linger_Errors.err
#SBATCH -p compute
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu=50G

source /gpfs/Home/esm5360/.bashrc

conda activate LINGER_1.92

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/MESC_PIPELINE/

OUTDIR='/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_70_SUBSAMPLE'

# Decide which method to use (scNN or LINGER)
GENOME='mm10'
METHOD='scNN'
CELLTYPE='all'
ACTIVEF='ReLU' # Neural Network activation function. Choose from: 'ReLU','sigmoid','tanh'

# Main paths - these do not change
UZUN_LAB_DIR='/gpfs/Labs/Uzun'
DATA_DIR="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
GROUND_TRUTH_DIR="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
TSS_MOTIF_INFO_PATH="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"

# Output directory
OUTPUT_DIR="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_TRAINED_MODEL"

RESULTS_DIR="${UZUN_LAB_DIR}/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS"

# List of sample numbers to iterate through
SAMPLE_NUMS=(1000 2000 3000 4000 5000)

# Iterate through each sample number
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  # Define RNA and ATAC data paths for the current sample number
  RNA_DATA_PATH="${DATA_DIR}/subsampled_RNA_${SAMPLE_NUM}.csv"
  ATAC_DATA_PATH="${DATA_DIR}/subsampled_ATAC_${SAMPLE_NUM}.csv"
  
  # Create subdirectories in OUTPUT_DIR and RESULTS_DIR for the current sample number
  SAMPLE_DATA_DIR="${OUTPUT_DIR}/sample_${SAMPLE_NUM}"
  SAMPLE_OUTPUT_DIR="${OUTPUT_DIR}/sample_${SAMPLE_NUM}"
  SAMPLE_RESULTS_DIR="${RESULTS_DIR}/sample_${SAMPLE_NUM}"
  
  mkdir -p "${SAMPLE_DATA_DIR}"
  mkdir -p "${SAMPLE_OUTPUT_DIR}"
  mkdir -p "${SAMPLE_RESULTS_DIR}"
  
  # Print status
  echo "Processing sample number ${SAMPLE_NUM}..."
  echo "RNA data path: ${RNA_DATA_PATH}"
  echo "ATAC data path: ${ATAC_DATA_PATH}"
  echo "Output directory: ${SAMPLE_OUTPUT_DIR}"
  echo "Results directory: ${SAMPLE_RESULTS_DIR}"
  echo "Sample data directory: ${SAMPLE_DATA_DIR}"

  echo running Step_010.Linger_Load_Data.py
  
  python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_010.Linger_Load_Data.py \
  --rna_data_path "$RNA_DATA_PATH" \
  --atac_data_path "$ATAC_DATA_PATH" \
  --data_dir "$DATA_DIR" \
  --sample_data_dir "$SAMPLE_DATA_DIR" \
  --output_dir "$SAMPLE_OUTPUT_DIR"

  python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_020.Linger_Training.py \
  --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
  --genome "$GENOME" \
  --method "$METHOD" \
  --sample_data_dir "$SAMPLE_DATA_DIR" \
  --output_dir "$SAMPLE_OUTPUT_DIR" \
  --activef "$ACTIVEF"

  python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_030.Create_Cell_Population_GRN.py \
  --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
  --genome "$GENOME" \
  --method "$METHOD" \
  --sample_data_dir "$SAMPLE_DATA_DIR" \
  --output_dir "$SAMPLE_OUTPUT_DIR" \
  --activef "$ACTIVEF"

  python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_040.Homer_Motif_Finding.py \
  --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
  --sample_data_dir "$SAMPLE_DATA_DIR" \
  --genome "$GENOME" \
  --output_dir "$SAMPLE_OUTPUT_DIR"

  python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_030.Create_Cell_Population_GRN.py \
  --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
  --genome "$GENOME" \
  --method "$METHOD" \
  --sample_data_dir "$SAMPLE_DATA_DIR" \
  --output_dir "$SAMPLE_OUTPUT_DIR" \
  --celltype "$CELLTYPE"

done