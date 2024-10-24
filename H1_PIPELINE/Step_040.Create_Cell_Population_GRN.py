import scanpy as sc
import linger.LL_net as LL_net

import shared_variables

# Load in the adata_RNA and adata_ATAC files
print(f'Reading in the RNAseq and ATACseq h5ad adata')
adata_RNA = sc.read_h5ad(shared_variables.adata_RNA_outpath)
adata_ATAC = sc.read_h5ad(shared_variables.adata_ATAC_outpath)

# Calculate the TF RE binding potential
print(f'Calculating the TF RE binding potential')
LL_net.TF_RE_binding(
  shared_variables.bulk_model_dir,
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
  shared_variables.bulk_model_dir,
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
  shared_variables.bulk_model_dir,
  shared_variables.data_dir,
  shared_variables.method,
  shared_variables.output_dir,
  shared_variables.genome
  )