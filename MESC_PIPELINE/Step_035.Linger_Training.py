import sys
import pandas as pd

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import linger_1_92.LINGER_tr as LINGER_tr
import MESC_PIPELINE.shared_variables as shared_variables

print('Getting TSS')
LINGER_tr.get_TSS(
    shared_variables.tss_motif_info_path, 
    shared_variables.genome, 
    200000, # Here, 200000 represent the largest distance of regulatory element to the TG. Other distance is supported
    shared_variables.output_dir # I altered the function to allow for a different output directory
    ) 

print('Getting RE-TG distances')
LINGER_tr.RE_TG_dis(shared_variables.output_dir, shared_variables.data_dir)

genomemap=pd.read_csv(shared_variables.tss_motif_info_path+'genome_map_homer.txt',sep='\t')
genomemap.index=genomemap['genome_short']
species=genomemap.loc[shared_variables.genome]['species_ensembl']

# Refines the bulk model by further training it on the single-cell data
print(f'\nBeginning LINGER single cell training...')
LINGER_tr.training(
    shared_variables.tss_motif_info_path,
    shared_variables.method,
    shared_variables.output_dir,
    shared_variables.data_dir, # Altered the function to allow for the data dir to be separate from output_dir
    shared_variables.activef,
    species
    )

print(f'FINISHED TRAINING')