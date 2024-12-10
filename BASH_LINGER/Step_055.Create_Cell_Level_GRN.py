import scanpy as sc
import multiprocessing
import subprocess
import pandas as pd
import sys
import argparse

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


args = parser.parse_args()


def process_cells_in_parallel(cell_names_slice, adata_RNA, adata_ATAC, genome, output_dir, method, tss_motif_info_path):
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

def parallelize_cell_processing(adata_RNA, adata_ATAC, genome, output_dir, method, tss_motif_info_path, cell_names):
    """
    Function to split cells into chunks and process them in parallel using multiprocessing.
    """

    # Define the size of each chunk (e.g., process 10 cells at a time)
    chunk_size = 1  # You can adjust this depending on the number of cells and available resources

    # Split the cell names into chunks
    cell_name_chunks = [cell_names[i:i + chunk_size] for i in range(0, len(cell_names), chunk_size)]

    # Set up multiprocessing Pool to process each chunk of cells in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_cells_in_parallel, 
                     [(chunk, adata_RNA, adata_ATAC, genome, output_dir, method, tss_motif_info_path) for chunk in cell_name_chunks])



if args.organism.lower() == "mouse":
  import linger_1_92.LL_net as LL_net

  # Load in the adata_RNA and adata_ATAC files
  print(f'Reading in the RNAseq and ATACseq h5ad adata', flush=True)
  adata_RNA = sc.read_h5ad(f'{args.sample_data_dir}/adata_RNA.h5ad')
  adata_ATAC = sc.read_h5ad(f'{args.sample_data_dir}/adata_ATAC.h5ad')

  output_dir = args.sample_data_dir + "/"
  

  print(f'Calculating cell-type specific TF RE binding for celltype "{args.celltype}"', flush=True)
  LL_net.cell_type_specific_TF_RE_binding(
    args.tss_motif_info_path,
    adata_RNA,
    adata_ATAC,
    args.genome,
    args.celltype,
    output_dir,
    args.method,
    )

  print(f'Calculating cell-type specific cis-regulatory network for celltype "{args.celltype}"', flush=True)
  LL_net.cell_type_specific_cis_reg(
    args.tss_motif_info_path,
    adata_RNA,
    adata_ATAC,
    args.genome,
    args.celltype,
    output_dir,
    args.method
    )

  print(f'Calculating cell-type specific trans-regulatory network for celltype "{args.celltype}"', flush=True)
  LL_net.cell_type_specific_trans_reg(
    args.tss_motif_info_path,
    adata_RNA,
    args.celltype,
    output_dir,
    )

# Example usage in your script
if args.organism.lower() == "human":
    import linger.LL_net as LL_net
    # Load in the adata_RNA and adata_ATAC files
    print(f'Reading in the RNAseq and ATACseq h5ad adata', flush=True)
    adata_RNA = sc.read_h5ad(f'{args.sample_data_dir}/adata_RNA.h5ad')
    adata_ATAC = sc.read_h5ad(f'{args.sample_data_dir}/adata_ATAC.h5ad')

    output_dir = args.sample_data_dir + "/"

    print(f'Calculating cell level TF RE binding for celltype "{args.celltype}"', flush=True)
    
    cell_names = adata_RNA.obs_names.tolist()[0:5]
    
    # Call the function that parallelizes processing
    parallelize_cell_processing(
        adata_RNA, 
        adata_ATAC, 
        args.genome, 
        output_dir, 
        args.method,
        args.tss_motif_info_path, 
        cell_names
    )


  # print(f'Calculating cell-type specific cis-regulatory network for celltype "{args.celltype}"', flush=True)
  # LL_net.cell_type_specific_cis_reg(
  #   args.tss_motif_info_path,
  #   adata_RNA,
  #   adata_ATAC,
  #   args.genome,
  #   args.celltype,
  #   output_dir,
  #   args.method
  #   )

  # print(f'Calculating cell-type specific trans-regulatory network for celltype "{args.celltype}"', flush=True)
  # LL_net.cell_type_specific_trans_reg(
  #   args.tss_motif_info_path,
  #   adata_RNA,
  #   args.celltype,
  #   output_dir,
  #   )