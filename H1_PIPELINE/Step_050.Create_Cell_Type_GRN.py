import scanpy as sc
import linger.LL_net as LL_net

import shared_variables

# Load in the adata_RNA and adata_ATAC files
print(f'Reading in the RNAseq and ATACseq h5ad adata')
adata_RNA = sc.read_h5ad(shared_variables.adata_RNA_outpath)
adata_ATAC = sc.read_h5ad(shared_variables.adata_ATAC_outpath)

print(f'Calculating cell-type specific TF RE binding for celltype "{shared_variables.celltype}"')
LL_net.cell_type_specific_TF_RE_binding(
  shared_variables.bulk_model_dir,
  adata_RNA,
  adata_ATAC,
  shared_variables.genome,
  shared_variables.celltype,
  shared_variables.output_dir,
  shared_variables.method
  )

print(f'Calculating cell-type specific cis-regulatory network for celltype "{shared_variables.celltype}"')
LL_net.cell_type_specific_cis_reg(
  shared_variables.bulk_model_dir,
  adata_RNA,
  adata_ATAC,
  shared_variables.genome,
  shared_variables.celltype,
  shared_variables.output_dir,
  shared_variables.method
  )

print(f'Calculating cell-type specific trans-regulatory network for celltype "{shared_variables.celltype}"')
LL_net.cell_type_specific_trans_reg(
  shared_variables.bulk_model_dir,
  adata_RNA,
  shared_variables.celltype,
  shared_variables.output_dir,
  )
