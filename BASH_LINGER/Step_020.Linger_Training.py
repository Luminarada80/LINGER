import sys
import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import linger_1_92.LINGER_tr as LINGER_tr

parser = argparse.ArgumentParser(description="Train the scNN neural network model.")

# Add arguments for file paths and directories
parser.add_argument("--tss_motif_info_path", required=True, help="Path to the LINGER TSS information path for the organism")
parser.add_argument("--genome", required=True, help="Organism genome code")
parser.add_argument("--method", required=True, help="Training method")
parser.add_argument("--sample_data_dir", required=True, help="Directory containing LINGER intermediate files")
parser.add_argument("--output_dir", required=True, help="Output directory for results")
parser.add_argument("--activef", required=True, help="activation function to use for training")

args = parser.parse_args()

output_dir = args.sample_data_dir + "/"

logging.info('Getting TSS')
LINGER_tr.get_TSS(
    args.tss_motif_info_path,
    args.genome, 
    200000, # Here, 200000 represent the largest distance of regulatory element to the TG. Other distance is supported
    output_dir # I altered the function to allow for a different output directory
    ) 

logging.info('Getting RE-TG distances')
LINGER_tr.RE_TG_dis(output_dir, args.sample_data_dir)

genomemap=pd.read_csv(args.tss_motif_info_path+'genome_map_homer.txt',sep='\t')
genomemap.index=genomemap['genome_short']
species=genomemap.loc[args.genome]['species_ensembl']

# Refines the bulk model by further training it on the single-cell data
logging.info(f'\nBeginning LINGER single cell training...')
LINGER_tr.training(
    args.tss_motif_info_path,
    args.method,
    output_dir,
    args.sample_data_dir, # Altered the function to allow for the data dir to be separate from output_dir
    args.activef,
    species
    )

logging.info(f'FINISHED TRAINING')