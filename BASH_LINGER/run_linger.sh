#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem-per-cpu=16G

source /gpfs/Home/esm5360/.bashrc
conda activate LINGER_1.92
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/

LOG_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS"

# Dynamically set the job name
scontrol update JobID=$SLURM_JOB_ID JobName=linger_${SAMPLE_NUM}

GENOME='mm10'
METHOD='scNN'
CELLTYPE='all'
ACTIVEF='ReLU'

UZUN_LAB_DIR='/gpfs/Labs/Uzun'
DATA_DIR="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
TSS_MOTIF_INFO_PATH="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"

OUTPUT_DIR="${UZUN_LAB_DIR}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_TRAINED_MODEL"
RESULTS_DIR="${UZUN_LAB_DIR}/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS"

# Change these! Paths to the RNA and ATAC data
RNA_DATA_PATH="${DATA_DIR}/FULL_MESC_SAMPLES/multiomic_data_${SAMPLE_NUM}_RNA.csv"
ATAC_DATA_PATH="${DATA_DIR}/FULL_MESC_SAMPLES/multiomic_data_${SAMPLE_NUM}_ATAC.csv"
GROUND_TRUTH_DIR="${DATA_DIR}/RN111.tsv"

SAMPLE_DATA_DIR="${OUTPUT_DIR}/${SAMPLE_NUM}"
SAMPLE_OUTPUT_DIR="${OUTPUT_DIR}/${SAMPLE_NUM}/"
SAMPLE_RESULTS_DIR="${RESULTS_DIR}/${SAMPLE_NUM}"

mkdir -p "${SAMPLE_DATA_DIR}" "${SAMPLE_OUTPUT_DIR}" "${SAMPLE_RESULTS_DIR}" "${LOG_DIR}/${SAMPLE_NUM}/"

# Set output and error files dynamically
exec > "${LOG_DIR}/${SAMPLE_NUM}/Linger_Results_${SAMPLE_NUM}.txt" 2> "${LOG_DIR}/${SAMPLE_NUM}/Linger_Errors_${SAMPLE_NUM}.err"

echo "Processing sample number ${SAMPLE_NUM}..."

# Function to run each Python step with timing and memory tracking
run_step() {
  step_name=$1
  script_path=$2
  shift 2  # Remove the first two arguments
  echo "Running $step_name..."
  
  # Use /usr/bin/time with verbose output to capture memory and timing
  /usr/bin/time -v python3 "$script_path" "$@" 2>> "${LOG_DIR}/${SAMPLE_NUM}/${step_name}_time_mem.log"
}

# Run each step of the pipeline with resource tracking
# run_step "Step_010.Linger_Load_Data" "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_010.Linger_Load_Data.py" \
#   --rna_data_path "$RNA_DATA_PATH" \
#   --atac_data_path "$ATAC_DATA_PATH" \
#   --data_dir "$DATA_DIR" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR"

# run_step "Step_020.Linger_Training" "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_020.Linger_Training.py" \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --genome "$GENOME" \
#   --method "$METHOD" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR" \
#   --activef "$ACTIVEF"

# run_step "Step_030.Create_Cell_Population_GRN" "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_030.Create_Cell_Population_GRN.py" \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --genome "$GENOME" \
#   --method "$METHOD" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR" \
#   --activef "$ACTIVEF"

# run_step "Step_040.Homer_Motif_Finding" "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_040.Homer_Motif_Finding.py" \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --genome "$GENOME" \
#   --output_dir "$SAMPLE_OUTPUT_DIR"

# run_step "Step_050.Create_Cell_Type_GRN" "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_050.Create_Cell_Type_GRN.py" \
#   --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
#   --genome "$GENOME" \
#   --method "$METHOD" \
#   --sample_data_dir "$SAMPLE_DATA_DIR" \
#   --output_dir "$SAMPLE_OUTPUT_DIR" \
#   --celltype "$CELLTYPE"

run_step "Step_060.Analyze_Results" "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/Step_060.Analyze_Results.py" \
  --result_dir "$SAMPLE_RESULTS_DIR" \
  --output_dir "$SAMPLE_OUTPUT_DIR" \
  --ground_truth "$GROUND_TRUTH_DIR" \
  --cell_type "mESC" \
  --sample_num "$SAMPLE_NUM"
