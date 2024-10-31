#!/bin/bash

# Define sample numbers
SAMPLE_NUMS=(1000 2000 3000 4000 5000)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  sbatch --export=SAMPLE_NUM="$SAMPLE_NUM" run_linger.sh
done