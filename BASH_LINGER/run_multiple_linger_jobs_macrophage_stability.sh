#!/bin/bash

# Define sample numbers
# 1 2 3 4 5 6 7 8 9 10

# 1000 2000 3000 4000 5000




SAMPLE_NUMS=(
  "1"
  # "Macrophase_buffer1_stability_2"
  # "Macrophase_buffer1_stability_3" 
  # "Macrophase_buffer1_stability_4" 
  # "Macrophase_buffer1_stability_5" 
  # "Macrophase_buffer1_stability_6" 
  # "Macrophase_buffer1_stability_7" 
  # "Macrophase_buffer1_stability_8" 
  # "Macrophase_buffer1_stability_9" 
  # "Macrophase_buffer1_stability_10"
  # "Macrophase_buffer2_stability_1" 
  # "Macrophase_buffer2_stability_2" 
  # "Macrophase_buffer2_stability_3" 
  # "Macrophase_buffer2_stability_4" 
  # "Macrophase_buffer2_stability_5" 
  # "Macrophase_buffer2_stability_6" 
  # "Macrophase_buffer2_stability_7" 
  # "Macrophase_buffer2_stability_8" 
  # "Macrophase_buffer2_stability_9" 
  # "Macrophase_buffer2_stability_10"
  # "Macrophase_buffer3_stability_1" 
  # "Macrophase_buffer3_stability_2" 
  # "Macrophase_buffer3_stability_3" 
  # "Macrophase_buffer3_stability_4" 
  # "Macrophase_buffer3_stability_5" 
  # "Macrophase_buffer3_stability_6" 
  # "Macrophase_buffer3_stability_7" 
  # "Macrophase_buffer3_stability_8" 
  # "Macrophase_buffer3_stability_9" 
  # "Macrophase_buffer3_stability_10"
  # "Macrophase_buffer4_stability_1" 
  # "Macrophase_buffer4_stability_2" 
  # "Macrophase_buffer4_stability_3" 
  # "Macrophase_buffer4_stability_4" 
  # "Macrophase_buffer4_stability_5" 
  # "Macrophase_buffer4_stability_6" 
  # "Macrophase_buffer4_stability_7" 
  # "Macrophase_buffer4_stability_8" 
  # "Macrophase_buffer4_stability_9" 
  # "Macrophase_buffer4_stability_10"
)

# Submit each SAMPLE_NUM as a separate job
for SAMPLE_NUM in "${SAMPLE_NUMS[@]}"; do
  sbatch --export=SAMPLE_NUM="$SAMPLE_NUM" run_linger_macrophage_stability.sh
done