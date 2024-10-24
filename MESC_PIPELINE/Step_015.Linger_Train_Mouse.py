import pandas as pd
import LingerGRN.LINGER_tr as LINGER_tr
import linger.LINGER_tr as my_linger_tr

import MESC_PIPELINE.shared_variables as shared_variables

GRNdir = shared_variables.tss_motif_info_path# This directory should be the same as Datadir defined in the above 'Download the general gene regulatory network' section
genome = shared_variables.genome
outdir = shared_variables.output_dir #output dir
activef = shared_variables.activef
method = shared_variables.method

print('Getting TSS')
LINGER_tr.get_TSS(GRNdir, genome, 200000) # Here, 200000 represent the largest distance of regulatory element to the TG. Other distance is supported

print('Getting RE-TG distances')
LINGER_tr.RE_TG_dis(shared_variables.data_dir)

activef='ReLU' # active function chose from 'ReLU','sigmoid','tanh'
genomemap=pd.read_csv(GRNdir+'genome_map_homer.txt',sep='\t')
genomemap.index=genomemap['genome_short']
species=genomemap.loc[genome]['species_ensembl']

print('Training using scNN')
LINGER_tr.training(GRNdir,method,outdir,activef,species)