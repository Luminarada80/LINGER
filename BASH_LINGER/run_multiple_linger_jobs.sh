#!/bin/bash

# Define sample numbers
SAMPLE_NUMS=(1 2 3 4 5 6 7 8 9 10)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  sbatch --export=SAMPLE_NUM="$SAMPLE_NUM" run_linger.sh
done