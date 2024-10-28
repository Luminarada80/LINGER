import scanpy as sc

import sys
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import linger_1_92.LL_net as LL_net

import MESC_PIPELINE.shared_variables as shared_variables

# Load in the adata_RNA and adata_ATAC files
print(f'Reading in the RNAseq and ATACseq h5ad adata')
adata_RNA = sc.read_h5ad(f'{shared_variables.data_dir}/adata_RNA.h5ad')
adata_ATAC = sc.read_h5ad(f'{shared_variables.data_dir}/adata_ATAC.h5ad')

# Calculate the TF RE binding potential
print(f'Calculating the TF RE binding potential')
LL_net.TF_RE_binding(
  shared_variables.tss_motif_info_path,
  shared_variables.data_dir,
  adata_RNA,
  adata_ATAC,
  shared_variables.genome,
  shared_variables.method,
  shared_variables.output_dir
  )

# Calculate the cis-regulatory scores
print(f'Calculating the cis-regulatory network')
LL_net.cis_reg(
  shared_variables.tss_motif_info_path,
  shared_variables.data_dir,
  adata_RNA,
  adata_ATAC,
  shared_variables.genome,
  shared_variables.method,
  shared_variables.output_dir
  )

# Calculate the trans-regulatory scores
print(f'Calculating the trans-regulatory network')
LL_net.trans_reg(
  shared_variables.tss_motif_info_path,
  shared_variables.data_dir,
  shared_variables.method,
  shared_variables.output_dir,
  shared_variables.genome
  )