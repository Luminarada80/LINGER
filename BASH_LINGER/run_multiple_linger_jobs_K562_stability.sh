#!/bin/bash

# Define sample numbers

SAMPLE_NUMS=(
  # K562_stability_1
  K562_stability_2
  K562_stability_3
  K562_stability_4
  K562_stability_5
  K562_stability_6
  K562_stability_7
  K562_stability_8
  K562_stability_9
  K562_stability_10
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  sbatch --export=SAMPLE_NUM="$SAMPLE_NUM" run_linger_K562_stability.sh
done