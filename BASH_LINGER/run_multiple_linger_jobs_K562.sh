#!/bin/bash

# Define sample numbers

SAMPLE_NUMS=(
  "K562_human_filtered"
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  sbatch --export=SAMPLE_NUM="$SAMPLE_NUM" run_linger_K562.sh
done