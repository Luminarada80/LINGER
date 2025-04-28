#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem 64G

set -euo pipefail

# ==========================================
#             USER VARIABLES
# ==========================================

# Conda environment name
CONDA_ENV_NAME="LINGER"

GENOME='mm10'
METHOD='scNN'
CELLTYPE='all'
ACTIVEF='ReLU'
ORGANISM="mouse"

# Scripts and data paths
SCRIPTS_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER"
DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA"
RESULTS_DIR="/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/mESC_RESULTS"
BULK_MODEL_DIR=""

# Sample-specific variables (you must export SAMPLE_NUM before running)
RNA_DATA_PATH="${DATA_DIR}/FULL_MESC_SAMPLES/${SAMPLE_NUM}/multiomic_data_${SAMPLE_NUM}_RNA.csv"
ATAC_DATA_PATH="${DATA_DIR}/FULL_MESC_SAMPLES/${SAMPLE_NUM}/multiomic_data_${SAMPLE_NUM}_ATAC.csv"
GROUND_TRUTH_PATH="${DATA_DIR}/RN111.tsv"

# Motif and TSS information for non-human samples (for Homer)
TSS_MOTIF_INFO_PATH="${DATA_DIR}/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/"

# Paths in the results directory
SAMPLE_RESULTS_DIR="${RESULTS_DIR}/FULL_MESC_SAMPLES/${SAMPLE_NUM}"
SAMPLE_DATA_DIR="${DATA_DIR}/${SAMPLE_NUM}"

LOG_DIR="${SCRIPTS_DIR}/LOGS"

# ==========================================
#             SETUP FUNCTIONS
# ==========================================

validate_critical_variables() {
    echo "[INFO] Validating required variables..."
    local required_vars=(
        SAMPLE_NUM
        RNA_DATA_PATH
        ATAC_DATA_PATH
        GROUND_TRUTH_PATH
        SCRIPTS_DIR
        DATA_DIR
        RESULTS_DIR
    )
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            echo "[ERROR] Required variable '$var' is not set."
            exit 1
        fi
    done
}

check_for_running_jobs() {
    echo "[INFO] Checking for running jobs with the same name..."
    if [ -n "${SLURM_JOB_NAME:-}" ]; then
        local running_jobs
        running_jobs=$(squeue --name="$SLURM_JOB_NAME" --noheader | wc -l)
        if [ "$running_jobs" -gt 1 ]; then
            echo "[ERROR] Another job with the same name is already running. Exiting."
            exit 1
        else
            echo "[INFO] No conflicting jobs detected."
        fi
    else
        echo "[INFO] Not running under SLURM, skipping job check."
    fi
}

check_tools() {
    echo "[INFO] Checking for required tools (python3, conda)..."
    for tool in python3 conda; do
        if ! command -v "$tool" &> /dev/null; then
            echo "[ERROR] Required tool '$tool' is not installed or not in PATH."
            exit 1
        else
            echo "    - Found $tool"
        fi
    done
}

activate_conda_env() {
    echo "[INFO] Activating Conda environment '$CONDA_ENV_NAME'..."
    if ! conda activate "$CONDA_ENV_NAME"; then
        echo "[ERROR] Could not activate Conda environment '$CONDA_ENV_NAME'."
        exit 1
    fi
    echo "    - Successfully activated Conda environment."
}

check_input_files() {
    echo "[INFO] Checking if input RNA and ATAC files exist..."
    for file in "$RNA_DATA_PATH" "$ATAC_DATA_PATH"; do
        if [ ! -f "$file" ]; then
            echo "[ERROR] Missing input file: $file"
            exit 1
        else
            echo "    - Found $file"
        fi
    done
}

setup_directories() {
    echo "[INFO] Setting up necessary directories..."
    mkdir -p "$SAMPLE_RESULTS_DIR" "$SAMPLE_DATA_DIR"
    echo "    - Directories created."
}

set_slurm_job_name() {
    echo "[INFO] Setting dynamic SLURM job name..."
    scontrol update JobID="$SLURM_JOB_ID" JobName="LINGER_${SAMPLE_NUM}"
}

# ==========================================
#             PIPELINE STEPS
# ==========================================

# Function to run each Python step with timing and memory tracking
run_step() {
  step_name=$1
  script_path=$2
  shift 2  # Remove the first two arguments
  echo "Running $step_name..."
  
  # Use /usr/bin/time with verbose output to capture memory and timing
  /usr/bin/time -v python3 "$script_path" "$@" 2> "${LOG_DIR}/${SAMPLE_NUM}/${step_name}_time_mem.log"
}

run_pipeline() {
  # #Run each step of the pipeline with resource tracking
  run_step "Step_010.Linger_Load_Data" "${SCRIPTS_DIR}/Step_010.Linger_Load_Data.py" \
    --rna_data_path "$RNA_DATA_PATH" \
    --atac_data_path "$ATAC_DATA_PATH" \
    --data_dir "$DATA_DIR" \
    --sample_data_dir "$SAMPLE_DATA_DIR" \
    --organism "$ORGANISM" \
    --bulk_model_dir "$BULK_MODEL_DIR" \
    --genome "$GENOME" \
    --method "$METHOD"

  run_step "Step_020.Linger_Training" "${SCRIPTS_DIR}/Step_020.Linger_Training.py" \
    --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
    --genome "$GENOME" \
    --method "$METHOD" \
    --sample_data_dir "$SAMPLE_DATA_DIR" \
    --activef "$ACTIVEF" \
    --organism "$ORGANISM" \
    --bulk_model_dir "$BULK_MODEL_DIR"

  run_step "Step_030.Create_Cell_Population_GRN" "${SCRIPTS_DIR}/Step_030.Create_Cell_Population_GRN.py" \
    --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
    --genome "$GENOME" \
    --method "$METHOD" \
    --sample_data_dir "$SAMPLE_DATA_DIR" \
    --activef "$ACTIVEF" \
    --organism "$ORGANISM" 

  run_step "Step_040.Homer_Motif_Finding" "${SCRIPTS_DIR}/Step_040.Homer_Motif_Finding.py" \
    --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
    --sample_data_dir "$SAMPLE_DATA_DIR" \
    --genome "$GENOME"

  run_step "Step_050.Create_Cell_Type_GRN" "${SCRIPTS_DIR}/Step_050.Create_Cell_Type_GRN.py" \
    --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
    --genome "$GENOME" \
    --method "$METHOD" \
    --sample_data_dir "$SAMPLE_DATA_DIR" \
    --celltype "$CELLTYPE" \
    --organism "$ORGANISM"

  run_step "Step_055.Create_Cell_Level_GRN.py" "${SCRIPTS_DIR}/Step_055.Create_Cell_Level_GRN.py" \
    --tss_motif_info_path "$TSS_MOTIF_INFO_PATH" \
    --genome "$GENOME" \
    --method "$METHOD" \
    --sample_data_dir "$SAMPLE_DATA_DIR" \
    --celltype "$CELLTYPE" \
    --organism "$ORGANISM"
}

# ==========================================
#               MAIN
# ==========================================


validate_critical_variables
check_for_running_jobs
check_tools
check_input_files
activate_conda_env
setup_directories
set_slurm_job_name
run_pipeline
