import scanpy as sc
import subprocess
import pandas as pd
import sys
import argparse

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import linger_1_92.LL_net as LL_net

parser = argparse.ArgumentParser(description="Train the scNN neural network model.")

# Add arguments for file paths and directories
parser.add_argument("--tss_motif_info_path", required=True, help="Path to the LINGER TSS information path for the organism")
parser.add_argument("--genome", required=True, help="Organism genome code")
parser.add_argument("--method", required=True, help="Training method")
parser.add_argument("--sample_data_dir", required=True, help="Directory containing LINGER intermediate files")
parser.add_argument("--output_dir", required=True, help="Output directory for results")
parser.add_argument("--celltype", required=True, help="Cell type for calculating cell-type specific GRNs")

args = parser.parse_args()

# Load in the adata_RNA and adata_ATAC files
print(f'Reading in the RNAseq and ATACseq h5ad adata')
adata_RNA = sc.read_h5ad(f'{args.sample_data_dir}/adata_RNA.h5ad')
adata_ATAC = sc.read_h5ad(f'{args.sample_data_dir}/adata_ATAC.h5ad')

print(f'Calculating cell-type specific TF RE binding for celltype "{args.celltype}"')
LL_net.cell_type_specific_TF_RE_binding(
  args.tss_motif_info_path,
  adata_RNA,
  adata_ATAC,
  args.genome,
  args.celltype,
  args.output_dir,
  args.method
  )

print(f'Calculating cell-type specific cis-regulatory network for celltype "{args.celltype}"')
LL_net.cell_type_specific_cis_reg(
  args.tss_motif_info_path,
  adata_RNA,
  adata_ATAC,
  args.genome,
  args.celltype,
  args.output_dir,
  args.method
  )

print(f'Calculating cell-type specific trans-regulatory network for celltype "{args.celltype}"')
LL_net.cell_type_specific_trans_reg(
  args.tss_motif_info_path,
  adata_RNA,
  args.celltype,
  args.output_dir,
  )