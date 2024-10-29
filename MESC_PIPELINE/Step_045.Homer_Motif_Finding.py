import subprocess
import pandas as pd
import sys

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import linger_1_92.LL_net as LL_net
import MESC_PIPELINE.shared_variables as shared_variables

command=f'paste {shared_variables.data_dir}/Peaks.bed {shared_variables.data_dir}/Peaks.txt > {shared_variables.data_dir}/region.txt'
subprocess.run(command, shell=True)

genome_map=pd.read_csv(shared_variables.tss_motif_info_path+'genome_map_homer.txt',sep='\t',header=0)
genome_map.index=genome_map['genome_short']
command=f'findMotifsGenome.pl {shared_variables.data_dir}/region.txt '+'mm10'+' ./. -size given -find '+shared_variables.tss_motif_info_path+'all_motif_rmdup_'+genome_map.loc[shared_variables.genome]['Motif']+'> '+shared_variables.output_dir+'MotifTarget.bed'
subprocess.run(command, shell=True)