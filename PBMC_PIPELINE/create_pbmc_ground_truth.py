import pandas as pd    
import os

pbmc_ground_truth_dir = '/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_PBMC_CISTROME'

cell_type_dict = {
    5967: 'naive B cells',
    8481: 'myeloid DC',
    40215: 'naive B cells',
    41287: 'classical monocytes',
    41288: 'classical monocytes',
    41289: 'classical monocytes',
    41290: 'classical monocytes',
    41301: 'classical monocytes',
    41302: 'classical monocytes',
    41303: 'classical monocytes',
    44092: 'naive CD4 T cells',
    44093: 'naive CD4 T cells',
    44094: 'naive CD4 T cells',
    44097: 'naive CD4 T cells',
    44098: 'naive CD4 T cells',
    45178: 'naive B cells',
    45444: 'classical monocytes',
    47435: 'naive CD4 T cells',
    81223: 'classical monocytes',
    85986: 'classical monocytes'
}

tf_list = ['CTCF', 'ETS1', 'FOXP3', 'IRF1', 'IRF4', 'MYC', 'REST', 'RUNX1', 'SPI1', 'STAT1']

# Initialize an empty dictionary to store results by filename
data_dict = {}

# List to store filenames for ground truth data
ground_truth_filenames = []

for file in os.listdir(pbmc_ground_truth_dir):
    print(f'Ground truth file: {file}')
    
    if file.endswith('.txt'):
        # Add the filename to the list
        ground_truth_filenames.append(f'{pbmc_ground_truth_dir}/{file}')

for idx, tf in enumerate(tf_list):
    tf_ground_truth_files = [file for file in ground_truth_filenames if tf in file ]
    for file in tf_ground_truth_files:
        # Load ground truth data, skipping the first 5 rows for header information
        ground_truth_data = pd.read_csv(f'{file}', sep='\t', skiprows=5, header=0)
        
        # Group ground truth by target gene ('symbol') and get the highest 'score' for each gene
        highest_scores = ground_truth_data.groupby('symbol')['score'].max()
        
        # Sort the target genes by score in descending order
        sorted_ground_truth = highest_scores.sort_values(ascending=False).reset_index()

        # Select the top 1000 target genes as ground truth
        top_n = 1000
        top_target_genes = sorted_ground_truth['symbol'].iloc[:top_n]

        # Add the top 1000 genes as targets of the TF
        data_dict[tf] = top_target_genes
        
ground_truth_df = pd.DataFrame(data_dict)

with open(f'{pbmc_ground_truth_dir}/PBMC_cistrome_ground_truth.csv', 'w') as outfile:
    for tf in ground_truth_df.columns:
        for gene in ground_truth_df[tf]:
            outfile.write(f"{tf} {gene}\n")



