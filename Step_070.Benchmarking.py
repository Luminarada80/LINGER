import logging
import os   
from linger import Benchmk
import shared_variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


RESULT_DIR: str = '/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER'
AUC_AUPR_OUTPUT_DIR: str = f'{RESULT_DIR}/AUC_AUPR'
os.makedirs(f'{AUC_AUPR_OUTPUT_DIR}/', exist_ok=True)

# Reset the auc and aupr files
with open(f'{AUC_AUPR_OUTPUT_DIR}/auc_scores.txt', 'w') as auc_file, open(f'{AUC_AUPR_OUTPUT_DIR}/aupr_scores.txt', 'w') as aupr_file:
    pass

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

ground_truth_filenames = []
for file in os.listdir(shared_variables.ground_truth_dir):
    print(f'Ground truth file: {file}')
    if file.endswith('.txt'):
        ground_truth_filenames.append(f'{shared_variables.ground_truth_dir}/{file}')

# Calculate the AUC and AUPR using LINGER Benchmk
logging.info(f'\n----- AUC and AUPR -----')
for idx, tf in enumerate(tf_list):
    tf_ground_truth_files = [file for file in ground_truth_filenames if tf in file ]
    for file in tf_ground_truth_files:
        cistrome_db = int(file.split('_')[-2].split('/')[-1])
        cell_type = cell_type_dict[cistrome_db]
        logging.info(f'\t{tf} ({idx+1}/{len(set(tf_list))})')
        tf_name = tf
        cell_types=[cell_type]
        predicted_interactions=[f'{shared_variables.output_dir}cell_type_specific_trans_regulatory_{cell_type}.txt']
        ground_truth_file=file
        output_dir=f'{AUC_AUPR_OUTPUT_DIR}/'
        data_type='matrix'
        Benchmk.evaluate_transcription_predictions(
            tf_name,
            cell_types,
            ground_truth_file, 
            predicted_interactions, 
            output_dir, 
            data_type
            )