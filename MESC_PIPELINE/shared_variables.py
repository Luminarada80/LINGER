# Main paths
uzun_lab_dir: str = '/gpfs/Labs/Uzun'
data_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA'
ground_truth_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA'

# PBMC file paths
rna_data_path: str = f'{data_dir}/subsampled_RNA_1000.csv'
atac_data_path: str = f'{data_dir}/subsampled_ATAC_1000.csv'

tss_motif_info_path: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_OTHER_SPECIES_TF_MOTIF_DATA/provide_data/'

# Output directory
output_dir: str = f'{uzun_lab_dir}/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_TRAINED_MODEL/'

# Decide which method to use (scNN or LINGER)
genome: str ='mm10'
method: str = 'scNN'
celltype: str = 'all'
activef: str = 'ReLU' # Neural Network activation function. Choose from: 'ReLU','sigmoid','tanh'