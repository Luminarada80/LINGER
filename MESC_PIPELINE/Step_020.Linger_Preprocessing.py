import pandas as pd
import sys

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

from linger.preprocess import *

import MESC_PIPELINE.shared_variables as shared_variables

# Read in the pseudobulk data and the peak file
print('Reading in pseudobulk and peak files')
TG_pseudobulk = pd.read_csv(shared_variables.TG_pseudobulk_path, sep='\t', index_col=0)
RE_pseudobulk = pd.read_csv(shared_variables.RE_pseudobulk_path, sep='\t', index_col=0)

# Overlap the region with the general GRN
print('Overlapping the regions with the general model')
preprocess(
    TG_pseudobulk,
    RE_pseudobulk,
    shared_variables.peak_file_path,
    shared_variables.bulk_model_dir,
    shared_variables.genome,
    shared_variables.method,
    shared_variables.output_dir
    )

print('Finished Preprocessing')

