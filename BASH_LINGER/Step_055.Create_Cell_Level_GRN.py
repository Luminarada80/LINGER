import scanpy as sc
import multiprocessing
from tqdm import tqdm
import logging, sys, os
import argparse
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import linger.LL_net as LL_net

parser = argparse.ArgumentParser(description="Train the scNN neural network model.")

# Add arguments for file paths and directories
parser.add_argument("--tss_motif_info_path", required=True, help="Path to the LINGER TSS information path for the organism")
parser.add_argument("--genome", required=True, help="Organism genome code")
parser.add_argument("--method", required=True, help="Training method")
parser.add_argument("--sample_data_dir", required=True, help="Directory containing LINGER intermediate files")
parser.add_argument("--celltype", required=True, help="Cell type for calculating cell-type specific GRNs")
parser.add_argument("--organism", required=True, help='Enter "mouse" or "human"')
parser.add_argument("--num_cpus", required=True, help='Number of cpus allocated for the job')
parser.add_argument("--num_cells", required=True, help='Number of cells to generate GRNs for')

args = parser.parse_args()

def _init_worker(rna, atac, genome, method, tss_info, out_dir, tf_binding):
    global _adata_RNA, _adata_ATAC, _genome, _method, _tss_info, _out_dir, _tf_binding
    _adata_RNA  = rna
    _adata_ATAC = atac
    _genome     = genome
    _method     = method
    _tss_info   = tss_info
    _out_dir    = out_dir
    _tf_binding = tf_binding

def _process_chunk(chunk):
    try:
        try:
            # TF-RE binding
            LL_net.cell_level_TF_RE_binding(
                _tss_info, _adata_RNA, _adata_ATAC, _genome,
                chunk, _out_dir, _method, _tf_binding
            )
        except Exception as e:
            logging.error(f'Cell level TF-RE binding failed for cells {chunk}: {e}')
        
        try:
            # cis‑reg
            LL_net.cell_level_cis_reg(
                _tss_info, _adata_RNA, _adata_ATAC, _genome,
                chunk, _out_dir, _method
            )
        except Exception as e:
            logging.error(f'Cell level cis-reg binding failed for cells {chunk}: {e}')
        
        try:
            # trans‑reg
            LL_net.cell_level_trans_reg(chunk, _out_dir)
        except Exception as e:
            logging.error(f'Cell level trans-reg binding failed for cells {chunk}: {e}')
        
        return len(chunk)  # or anything you like
    except Exception as e:
        logging.error(f"chunk {chunk} crashed: {e}")
    return len(chunk)

# Load in the adata_RNA and adata_ATAC files
logging.info(f'Reading in the RNAseq and ATACseq h5ad adata')
adata_RNA = sc.read_h5ad(f'{args.sample_data_dir}/adata_RNA.h5ad')
adata_ATAC = sc.read_h5ad(f'{args.sample_data_dir}/adata_ATAC.h5ad')

output_dir = args.sample_data_dir

cell_outdir = os.path.join(output_dir, "CELL_SPECIFIC_GRNS")
if not os.path.exists(cell_outdir):
    os.makedirs(cell_outdir)

logging.info(f'Calculating cell level GRNs for celltype "{args.celltype}"')

num_cells = args.num_cells
cell_names_all = adata_RNA.obs_names.tolist()
cell_names = random.sample(cell_names_all, min(len(cell_names_all), int(num_cells)))

# Define the size of each chunk (e.g., process 10 cells at a time)
chunk_size = 10  # You can adjust this depending on the number of cells and available resources

# Split the cell names into chunks
cell_name_chunks = [cell_names[i:i + chunk_size] for i in range(0, len(cell_names), chunk_size)]

num_cpus = int(args.num_cpus)

if args.genome == "mm10":
    TFbinding = LL_net.load_TFbinding_scNN(args.tss_motif_info_path, output_dir, args.genome)
else:
    TFbinding = None

logging.info(f"\nLaunching processing on {num_cpus} workers…")
with multiprocessing.Pool(
    processes=num_cpus,
    initializer=_init_worker,
    initargs=(adata_RNA, adata_ATAC, args.genome,
                args.method, args.tss_motif_info_path,
                args.sample_data_dir, TFbinding)
) as pool:
    for processed in tqdm(
        pool.imap_unordered(_process_chunk, cell_names),
        total=len(cell_names),
        desc="Cells",
        unit="cell"
    ):
        logging.info(f"  • Finished chunk of {processed} cells")

logging.info("All chunks complete")