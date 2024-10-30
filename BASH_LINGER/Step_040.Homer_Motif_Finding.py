import subprocess
import pandas as pd
import sys
import argparse

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import linger_1_92.LL_net as LL_net

parser = argparse.ArgumentParser(description="Train the scNN neural network model.")

parser.add_argument("--tss_motif_info_path", required=True, help="Path to the LINGER TSS information path for the organism")
parser.add_argument("--sample_data_dir", required=True, help="Directory containing LINGER intermediate files")
parser.add_argument("--genome", required=True, help="Organism genome code")
parser.add_argument("--output_dir", required=True, help="Output directory for results")

args = parser.parse_args()


command=f'paste {args.sample_data_dir}/Peaks.bed {args.sample_data_dir}/Peaks.txt > {args.sample_data_dir}/region.txt'
subprocess.run(command, shell=True)

genome_map=pd.read_csv(args.tss_motif_info_path+'genome_map_homer.txt',sep='\t',header=0)
genome_map.index=genome_map['genome_short']
command=f'findMotifsGenome.pl {args.sample_data_dir}/region.txt '+'mm10'+' ./. -size given -find '+args.tss_motif_info_path+'all_motif_rmdup_'+genome_map.loc[args.genome]['Motif']+'> '+args.output_dir+'MotifTarget.bed'
subprocess.run(command, shell=True)