#!/bin/bash

# Define sample numbers
# 1 2 3 4 5 6 7 8 9 10

# 1000 2000 3000 4000 5000
SAMPLE_NUMS=(1000 2000 3000 4000 5000)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  sbatch --export=SAMPLE_NUM="$SAMPLE_NUM" run_linger.sh
done