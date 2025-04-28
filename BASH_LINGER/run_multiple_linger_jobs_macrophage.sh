#!/bin/bash

SAMPLE_NUMS=(
  "Macrophase_buffer1_filtered"
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do

  mkdir -p "LOGS/${SAMPLE_NUM}"
  sbatch \
    --export=SAMPLE_NUM="$SAMPLE_NUM" \
    --output="LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.out" \
    --error="LOGS/${SAMPLE_NUM}/${SAMPLE_NUM}.err" \
    --job-name="LINGER_${SAMPLE_NUM}" \
    run_linger_macrophage.sh
done