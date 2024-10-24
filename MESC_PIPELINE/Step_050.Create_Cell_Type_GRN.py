import scanpy as sc
import LingerGRN.LL_net as LL_net
import subprocess
import pandas as pd

import MESC_PIPELINE.shared_variables as shared_variables

# Load in the adata_RNA and adata_ATAC files
print(f'Reading in the RNAseq and ATACseq h5ad adata')
adata_RNA = sc.read_h5ad(shared_variables.adata_RNA_outpath)
adata_ATAC = sc.read_h5ad(shared_variables.adata_ATAC_outpath)

command='paste data/Peaks.bed data/Peaks.txt > data/region.txt'
subprocess.run(command, shell=True)

genome_map=pd.read_csv(shared_variables.tss_motif_info_path+'genome_map_homer.txt',sep='\t',header=0)
genome_map.index=genome_map['genome_short']
command='findMotifsGenome.pl data/region.txt '+'mm10'+' ./. -size given -find '+shared_variables.tss_motif_info_path+'all_motif_rmdup_'+genome_map.loc[shared_variables.genome]['Motif']+'> '+shared_variables.output_dir+'MotifTarget.bed'
subprocess.run(command, shell=True)

print(f'Calculating cell-type specific TF RE binding for celltype "{shared_variables.celltype}"')
LL_net.cell_type_specific_TF_RE_binding(
  shared_variables.tss_motif_info_path,
  adata_RNA,
  adata_ATAC,
  shared_variables.genome,
  shared_variables.celltype,
  shared_variables.output_dir,
  shared_variables.method
  )

print(f'Calculating cell-type specific cis-regulatory network for celltype "{shared_variables.celltype}"')
LL_net.cell_type_specific_cis_reg(
  shared_variables.tss_motif_info_path,
  adata_RNA,
  adata_ATAC,
  shared_variables.genome,
  shared_variables.celltype,
  shared_variables.output_dir,
  shared_variables.method
  )

print(f'Calculating cell-type specific trans-regulatory network for celltype "{shared_variables.celltype}"')
LL_net.cell_type_specific_trans_reg(
  shared_variables.tss_motif_info_path,
  adata_RNA,
  shared_variables.celltype,
  shared_variables.output_dir,
  )
