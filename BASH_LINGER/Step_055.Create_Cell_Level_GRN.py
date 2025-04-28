import scanpy as sc
import multiprocessing
import subprocess
import pandas as pd
import sys
import argparse
import os
import random

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

parser = argparse.ArgumentParser(description="Train the scNN neural network model.")

# Add arguments for file paths and directories
parser.add_argument("--tss_motif_info_path", required=True, help="Path to the LINGER TSS information path for the organism")
parser.add_argument("--genome", required=True, help="Organism genome code")
parser.add_argument("--method", required=True, help="Training method")
parser.add_argument("--sample_data_dir", required=True, help="Directory containing LINGER intermediate files")
parser.add_argument("--celltype", required=True, help="Cell type for calculating cell-type specific GRNs")
parser.add_argument("--organism", required=True, help='Enter "mouse" or "human"')
parser.add_argument("--num_cells", required=True, help='Number of cells to generate GRNs for')


args = parser.parse_args()

def calculate_cell_level_TF_RE_in_parallel(cell_names_slice, adata_RNA, adata_ATAC, genome, output_dir, method, tss_motif_info_path):
    """
    Function to process a slice of cells in parallel.
    """
    print(f'Processing cells: {cell_names_slice}', flush=True)
    
    # Call the function to process a subset of cells
    LL_net.cell_level_TF_RE_binding(
        tss_motif_info_path,
        adata_RNA,
        adata_ATAC,
        genome,
        cell_names_slice,
        output_dir,
        method
    )

def calculate_cell_level_cis_reg_in_parallel(cell_names_slice, adata_RNA, adata_ATAC, genome, output_dir, method, tss_motif_info_path):
    """
    Function to process a slice of cells in parallel.
    """
    print(f'Processing cells: {cell_names_slice}', flush=True)
    
    LL_net.cell_level_cis_reg(
      tss_motif_info_path,
      adata_RNA,
      adata_ATAC,
      genome,
      cell_names_slice,
      output_dir,
      method
      )
    
def calculate_cell_level_trans_reg_in_parallel(cell_names_slice, output_dir):
    """
    Function to process a slice of cells in parallel.
    """
    print(f'Processing cells: {cell_names_slice}', flush=True)
    
    LL_net.cell_level_trans_reg(
      cell_names_slice,
      output_dir,
      )

# if args.organism.lower() == "mouse":
#   import linger_1_92.LL_net as LL_net

#   # Load in the adata_RNA and adata_ATAC files
#   print(f'Reading in the RNAseq and ATACseq h5ad adata', flush=True)
#   adata_RNA = sc.read_h5ad(f'{args.sample_data_dir}/adata_RNA.h5ad')
#   adata_ATAC = sc.read_h5ad(f'{args.sample_data_dir}/adata_ATAC.h5ad')

#   output_dir = args.sample_data_dir + "/CELL_SPECIFIC_GRNS/"

#   if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
  

#   print(f'Calculating cell-type specific TF RE binding for celltype "{args.celltype}"', flush=True)
#   LL_net.cell_type_specific_TF_RE_binding(
#     args.tss_motif_info_path,
#     adata_RNA,
#     adata_ATAC,
#     args.genome,
#     args.celltype,
#     output_dir,
#     args.method,
#     )

#   print(f'Calculating cell-type specific cis-regulatory network for celltype "{args.celltype}"', flush=True)
#   LL_net.cell_type_specific_cis_reg(
#     args.tss_motif_info_path,
#     adata_RNA,
#     adata_ATAC,
#     args.genome,
#     args.celltype,
#     output_dir,
#     args.method
#     )

#   print(f'Calculating cell-type specific trans-regulatory network for celltype "{args.celltype}"', flush=True)
#   LL_net.cell_type_specific_trans_reg(
#     args.tss_motif_info_path,
#     adata_RNA,
#     args.celltype,
#     output_dir,
#     )

# if args.organism.lower() == "human":
import linger.LL_net as LL_net
# Load in the adata_RNA and adata_ATAC files
print(f'Reading in the RNAseq and ATACseq h5ad adata', flush=True)
adata_RNA = sc.read_h5ad(f'{args.sample_data_dir}/adata_RNA.h5ad')
adata_ATAC = sc.read_h5ad(f'{args.sample_data_dir}/adata_ATAC.h5ad')

output_dir = args.sample_data_dir + "/CELL_SPECIFIC_GRNS/"

print(f'Calculating cell level TF RE binding for celltype "{args.celltype}"', flush=True)

num_cells = args.num_cells
cell_names_all = adata_RNA.obs_names.tolist()
cell_names = random.sample(cell_names_all, min(len(cell_names_all, num_cells)))

# Define the size of each chunk (e.g., process 10 cells at a time)
chunk_size = 10  # You can adjust this depending on the number of cells and available resources

# Split the cell names into chunks
cell_name_chunks = [cell_names[i:i + chunk_size] for i in range(0, len(cell_names), chunk_size)]

# Uses multiprocessing Pool to process each chunk of cells in parallel
print(f'{multiprocessing.cpu_count()} CPUs detected', flush=True)

# Calculate the cell-level TF-RE binding potential network
print(f'Calculating cell-level TF-RE binding potential network', flush=True)
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.starmap(calculate_cell_level_TF_RE_in_parallel, 
                  [(chunk, adata_RNA, adata_ATAC, args.genome, output_dir, args.method, args.tss_motif_info_path) for chunk in cell_name_chunks])

# Calculate the cell-level cis-regualtory binding potential network
print(f'Calculating cell-level cis-regulatory binding potential network', flush=True)
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.starmap(calculate_cell_level_cis_reg_in_parallel, 
                  [(chunk, adata_RNA, adata_ATAC, args.genome, output_dir, args.method, args.tss_motif_info_path) for chunk in cell_name_chunks])

# Calculate the cell-level trans-regulatory binding potential network
print(f'Calculating cell-level trans-regulatory binding potential network', flush=True)
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.starmap(calculate_cell_level_trans_reg_in_parallel, 
                  [(chunk, output_dir) for chunk in cell_name_chunks])

print("Done!", flush=True)