#!/bin/bash

# Define sample numbers
# 1 2 3 4 5 6 7 8 9 10

# 1000 2000 3000 4000 5000

# 
SAMPLE_NUMS=(
    "1000_cells_E7.5_rep1"
    "1000_cells_E7.5_rep2"
    "1000_cells_E7.75_rep1"
    "1000_cells_E8.0_rep1"
    "1000_cells_E8.0_rep2"
    "1000_cells_E8.5_CRISPR_T_KO"
    "1000_cells_E8.5_CRISPR_T_WT"
    "2000_cells_E7.5_rep1"
    "2000_cells_E8.0_rep1"
    "2000_cells_E8.0_rep2"
    "2000_cells_E8.5_CRISPR_T_KO"
    "2000_cells_E8.5_CRISPR_T_WT"
    "3000_cells_E7.5_rep1"
    "3000_cells_E8.0_rep1"
    "3000_cells_E8.0_rep2"
    "3000_cells_E8.5_CRISPR_T_KO"
    "3000_cells_E8.5_CRISPR_T_WT"
    "4000_cells_E7.5_rep1"
    "4000_cells_E8.0_rep1"
    "4000_cells_E8.0_rep2"
    "4000_cells_E8.5_CRISPR_T_KO"
    "4000_cells_E8.5_CRISPR_T_WT"
    "5000_cells_E7.5_rep1"
    "5000_cells_E8.5_CRISPR_T_KO"
    "5000_cells_E8.5_CRISPR_T_WT"
    "filtered_L2_E7.5_rep1"
    "filtered_L2_E7.5_rep2"
    "filtered_L2_E7.75_rep1"
    "filtered_L2_E8.0_rep1"
    "filtered_L2_E8.0_rep2"
    "filtered_L2_E8.5_CRISPR_T_KO"

)



# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  sbatch --export=SAMPLE_NUM="$SAMPLE_NUM" run_linger_mesc.sh
done