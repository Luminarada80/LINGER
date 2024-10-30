import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging
import os
import argparse
import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score

# Import the project directory to load the linger module
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

from linger import Benchmk
import MESC_PIPELINE.shared_variables as shared_variables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

CELL_TYPE = 'mESC' # H1

# ----- THESE VARIABLES NEED TO CHANGE DEPENDING ON DATASET -----
CHIP_SEQ_GROUND_TRUTH_PATH = f'{shared_variables.ground_truth_dir}/filtered_ground_truth_56TFs_3036TGs.csv'
RESULT_DIR: str = shared_variables.results_dir

# Set the value of the CELL_TYPE to 'all' if all TFs are in the cell line
CELL_TYPE_TF_DICT: dict = {
    'mESC': 'all'
}
# ----------------------------------------------------------------

# Allows the user to input whether they want to analyze the cell population or cell type results
def parse_args():
    parser = argparse.ArgumentParser(description="Process TF-TG pairs with cell type specificity.")
    parser.add_argument(
        "--cell_pop",
        action="store_true",
        help="Set CELL_POP to True if specified, otherwise False."
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        required=False,
        help="Enter the name of the cell type as it is formatted in the LINGER cell-type specific output directory"
    )

    args = parser.parse_args()

    # Default to cell population if no arguments passed
    if not args.cell_pop and args.cell_type is None:
        logging.warning("\n\nWarning! no arguments specified. Defaulting to running cell pop analysis\n")
        args.cell_pop = True

    # Check to make sure the entered cell type is in CELL_TYPE_TF_DICT
    if not args.cell_pop and not args.cell_type in CELL_TYPE_TF_DICT.keys():
        parser.error(f"\n\n--cell_type not found in the CELL_TYPE_TF_DICT: \n\n{CELL_TYPE_TF_DICT.keys()}")
    
    return args

# Parse the arguments
args = parse_args()
CELL_POP = args.cell_pop  # Default False
CELL_TYPE = args.cell_type

# Example usage
print(f'CELL_POP is set to "{CELL_POP}"')

if CELL_POP == False:
    print(f'CELL_TYPE is set to "{CELL_TYPE}"')

if CELL_POP == True:
    TF_RE_BINDING_PATH: str = f'{shared_variables.output_dir}cell_population_TF_RE_binding.txt'
    CIS_REG_NETWORK_PATH: str = f'{shared_variables.output_dir}cell_population_cis_regulatory.txt'
    TRANS_REG_NETWORK_PATH: str = f'{shared_variables.output_dir}cell_population_trans_regulatory.txt'

elif CELL_POP == False:
    TF_RE_BINDING_PATH: str = f'{shared_variables.output_dir}cell_type_specific_TF_RE_binding_{CELL_TYPE}.txt'
    CIS_REG_NETWORK_PATH: str = f'{shared_variables.output_dir}cell_type_specific_cis_regulatory_{CELL_TYPE}.txt'
    TRANS_REG_NETWORK_PATH: str = f'{shared_variables.output_dir}cell_type_specific_trans_regulatory_{CELL_TYPE}.txt'

    # Make sure that the cell type specific dataset is present
    os.makedirs(f'{RESULT_DIR}/{CELL_TYPE}', exist_ok=True)

else:
    logging.warning('CELL_POP is not set to boolean "True" or "False"')

# Sets the size of the text in the figures to be consistent
SMALL_SIZE: int = 8
MEDIUM_SIZE: int = 10
BIGGER_SIZE: int = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_data():
    """Load transcription factor (TF), cis-regulatory, and trans-regulatory network data."""
    logging.info("Loading TF-RE binding, cis-regulatory, and trans-regulatory network data.")
    tf_re_binding: pd.DataFrame = pd.read_csv(TF_RE_BINDING_PATH, sep='\t', index_col=0)
    cis_reg_network: pd.DataFrame = pd.read_csv(CIS_REG_NETWORK_PATH, sep='\t', index_col=0)
    trans_reg_network: pd.DataFrame = pd.read_csv(TRANS_REG_NETWORK_PATH, sep='\t', index_col=0)
    return tf_re_binding, cis_reg_network, trans_reg_network


def load_ground_truth():
    """Load ChIP-seq ground truth data."""
    logging.info("Loading ground truth data.")
    ground_truth: pd.DataFrame = pd.read_csv(CHIP_SEQ_GROUND_TRUTH_PATH, sep=',', header=0, index_col=0)
    return ground_truth


def process_tf_tg_pairs(ground_truth: pd.DataFrame, trans_reg_network: pd.DataFrame):
    """Process ground truth TF-TG pairs and retrieve the corresponding scores from the trans-regulatory network.
       Only considers TF-TG pairs for specified cell type TFs."""
    
    tf_list, tg_list, value_list = [], [], []
    num_nan: int = 0
    num_rows: int = 0

    for index, row in ground_truth.iterrows():
        try:
            tf, tg = row['Source'], row['Target']

            # The TF and TG names have to be capitalized like in the TRP score dataset
            tf = str(tf).capitalize()
            tg = str(tg).capitalize()

        # Catch the exception where the ground truth separator is different than the separatoor specified 
        # in load_ground_truth()
        except KeyError:
            logging.critical(
                f"\nKeyError at row {index} of ground_truth. Please check if the CSV file uses the correct delimiter "
                f"in 'load_ground_truth()'. \n\n Row: '{row}'"
            )
            sys.exit("Script terminated due to KeyError. Please resolve the issue and try again.")

        
        # If looking at a cell type, only use TFs specific to that cell type
        # Check if the TF is in the cell type-specific TF list
        if CELL_POP == False:

            # Finds only the TFs in the cell type or all of them if 'all' is the value for the cell type
            if tf in CELL_TYPE_TF_DICT[CELL_TYPE] or 'all' in CELL_TYPE_TF_DICT[CELL_TYPE]:
                try:
                    # Get the trans-regulatory score for the TF-TG pair
                    value = trans_reg_network.loc[tg.capitalize(), tf.capitalize()]
                    tf_list.append(tf)
                    tg_list.append(tg)
                    value_list.append(value)
                except KeyError:
                    # Handle missing value case
                    num_nan += 1
        
        # If using the cell population, use all TFs
        else:
            try:
                # Get the trans-regulatory score for the TF-TG pair
                value = trans_reg_network.loc[tg, tf]
                tf_list.append(tf)
                tg_list.append(tg)
                value_list.append(value)

            except KeyError:
                # Handle missing value case
                num_nan += 1

        
        num_rows += 1

    return tf_list, tg_list, value_list, num_nan


def save_ground_truth_scores(tf_list: list, tg_list: list, value_list: list):
    """Save the processed ground truth TF-TG scores to a CSV file."""
    # Convert the lists of TFs, TGs, and Scores to a dataframe
    ground_truth_df = pd.DataFrame({'TF': tf_list, 'TG': tg_list, 'Score': value_list})

    # Same the processed groudn truth dataframe to a csv file
    ground_truth_df.to_csv(f'{RESULT_DIR}/ground_truth_w_score.csv', header=True, index=False)
    logging.info("\nGround truth scores saved to 'ground_truth_w_score.csv'.")
    return ground_truth_df


def plot_trans_reg_distribution(trans_reg_network_minus_ground_truth: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """
    Plots a histogram of non-ground truth trans-regulatory potential scores compared to the ground truth scores.
    """

    linger_scores = trans_reg_network_minus_ground_truth['Score'].dropna()
    ground_truth_scores = ground_truth_df['Score'].dropna()

    # Plot histograms for the LINGER trans-reg network and ground truth network
    plt.hist(linger_scores, bins=150, log=True, alpha=0.7, label='Non-ground truth scores')
    plt.hist(ground_truth_scores, bins=150, log=True, alpha=0.7, label='Ground truth scores')

    
    plt.rc('')
    plt.ylabel('Frequency (log)')
    plt.xlabel('log2 LINGER trans-regulatory potential score')
    plt.legend()

    if CELL_POP == True:
        plt.title(f'Distribution of Cell Population TRP Scores')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/cell_pop_trans_reg_distribution.png', dpi=300)
    else:
        plt.title(f'Distribution of TRP Scores in {CELL_TYPE.capitalize()}')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/{CELL_TYPE}/trans_reg_distribution.png', dpi=300)
    plt.close()
    logging.info("\nTrans-regulatory distribution plot saved to 'trans_reg_distribution.png'.")

def plot_precision_recall_curve(
    ground_truth_df: pd.DataFrame, trans_reg_minus_ground_truth_df: pd.DataFrame
    ):
    # Combine ground truth and non-ground truth data for classification
    y_true = [1] * len(ground_truth_df) + [0] * len(trans_reg_minus_ground_truth_df)
    scores = np.concatenate([ground_truth_df['Score'], trans_reg_minus_ground_truth_df['Score']])

    # Compute precision-recall values
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid()

    # Save or show the plot
    save_path = f'{RESULT_DIR}/precision_recall_curve.png' if CELL_POP else f'{RESULT_DIR}/{CELL_TYPE}/precision_recall_curve.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    logging.info(f'Precision-Recall curve saved to {save_path}')
    plt.close()

def plot_trans_reg_distribution_with_thresholds(
    ground_truth_df: pd.DataFrame, trans_reg_minus_ground_truth_df: pd.DataFrame
):
    # Calculate thresholds
    lower_threshold = ground_truth_df['Score'].quantile(0.05)

    # Split data into TP, FP, TN, FN based on threshold
    tp_scores = ground_truth_df[ground_truth_df['Score'] >= lower_threshold]['Score']
    fn_scores = ground_truth_df[ground_truth_df['Score'] < lower_threshold]['Score']
    fp_scores = trans_reg_minus_ground_truth_df[trans_reg_minus_ground_truth_df['Score'] >= lower_threshold]['Score']
    tn_scores = trans_reg_minus_ground_truth_df[trans_reg_minus_ground_truth_df['Score'] < lower_threshold]['Score']

    # Plot histograms for each category with different colors
    plt.figure(figsize=(10, 6))
    plt.hist(tn_scores, bins=150, log=True, alpha=1, color='#b6cde0', label='True Negative (TN)')
    plt.hist(fp_scores, bins=150, log=True, alpha=1, color='#4195df', label='False Positive (FP)')
    plt.hist(fn_scores, bins=150, log=True, alpha=1, color='#efc69f', label='False Negative (FN)')
    plt.hist(tp_scores, bins=150, log=True, alpha=1, color='#dc8634', label='True Positive (TP)')

    # Plot threshold line
    plt.axvline(x=lower_threshold, color='black', linestyle='--', linewidth=2, label='Classification Threshold')

    # Add labels, title, and legend
    plt.xlabel('log2 LINGER trans-regulatory potential score')
    plt.ylabel('Frequency (log scale)')
    plt.title('Distribution of TRP Scores with Classification Labels')
    plt.legend(loc='upper left')

    # Save the plot
    if CELL_POP:
        plt.savefig(f'{RESULT_DIR}/cell_pop_trans_reg_distribution_classified.png', dpi=300)
    else:
        plt.savefig(f'{RESULT_DIR}/{CELL_TYPE}/trans_reg_distribution_classified.png', dpi=300)

    plt.close()
    logging.info("Trans-regulatory distribution plot with classification labels saved.")

def plot_box_whisker(trans_reg_network_minus_ground_truth: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """Create box and whisker plots to compare trans-regulatory network scores and ground truth scores."""
    # Handle zeros and small values by adding a small constant (1e-6)
    trans_reg_scores = trans_reg_network_minus_ground_truth['Score']
    ground_truth_scores = ground_truth_df['Score']

    # Apply log2 transformation after ensuring no zero or negative values
    lower_percentile=1
    upper_percentile=99

    # Remove outliers based on specified percentiles
    trans_reg_lower = np.percentile(trans_reg_scores, lower_percentile)
    trans_reg_upper = np.percentile(trans_reg_scores, upper_percentile)
    ground_truth_lower = np.percentile(ground_truth_scores, lower_percentile)
    ground_truth_upper = np.percentile(ground_truth_scores, upper_percentile)

    # Filter the data within the percentile range
    trans_reg_filtered = trans_reg_scores[(trans_reg_scores >= trans_reg_lower) & (trans_reg_scores <= trans_reg_upper)]
    ground_truth_filtered = ground_truth_scores[(ground_truth_scores >= ground_truth_lower) & (ground_truth_scores <= ground_truth_upper)]

    # Combine both into a single DataFrame for comparison
    scores_df = pd.DataFrame({
        'Score': np.concatenate([trans_reg_filtered, ground_truth_filtered]),
        'Source': ['True Negative'] * len(trans_reg_filtered) + ['True Positive (ground truth)'] * len(ground_truth_filtered)
    })

    # Plot the box and whisker plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Source', y='Score', data=scores_df)
    plt.ylabel('log2 Score')
    plt.xlabel('')
    
    plt.tight_layout()
    if CELL_POP == True:
        plt.title(f'Cell Population TRP Scores (no outliers)')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/cell_pop_trans_reg_box_whisker_plot_no_outliers.png', dpi=300)
    else:
        plt.title(f'TRP Scores in {CELL_TYPE.capitalize()} (no outliers)')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/{CELL_TYPE}/trans_reg_box_whisker_plot_no_outliers.png', dpi=300)

    plt.close()
    logging.info("Box and whisker plot saved to 'trans_reg_box_whisker_plot.png'.")


def plot_violin(trans_reg_network_minus_ground_truth: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """Create violin plots to compare trans-regulatory network scores and ground truth scores."""

    # Handle zeros and small values by adding a small constant (1e-6)
    trans_reg_scores = trans_reg_network_minus_ground_truth['Score']
    ground_truth_scores = ground_truth_df['Score']

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = trans_reg_scores
    ground_truth_scores = ground_truth_scores

    # Combine both into a single DataFrame for comparison
    scores_df = pd.DataFrame({
        'Score': np.concatenate([trans_reg_scores, ground_truth_scores]),
        'Source': ['True Negative'] * len(trans_reg_scores) + ['True Positive (ground truth)'] * len(ground_truth_scores)
    })

    # Plot the violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Source', y='Score', data=scores_df, inner="quartile")
    plt.ylabel('log2 Score')
    plt.xlabel('')
    
    if CELL_POP == True:
        plt.title(f'Cell Population TRP Scores')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/cell_pop_trans_reg_violin_plot.png', dpi=300)
    else:
        plt.title(f'{CELL_TYPE.capitalize()} TRP Scores')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/{CELL_TYPE}/trans_reg_violin_plot.png', dpi=300)

    plt.close()
    logging.info("Violin plot saved to 'trans_reg_violin_plot.png'.")


def plot_violin_without_outliers(trans_reg_network_minus_ground_truth: pd.DataFrame, ground_truth_df: pd.DataFrame, lower_percentile=1, upper_percentile=99):
    """Create violin plots for trans-regulatory network scores and ground truth scores after removing outliers."""

    # Handle zeros and small values by adding a small constant (1e-6)
    trans_reg_scores = trans_reg_network_minus_ground_truth['Score']
    ground_truth_scores = ground_truth_df['Score']

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = trans_reg_scores
    ground_truth_scores = ground_truth_scores

    # Remove outliers based on specified percentiles
    trans_reg_lower = np.percentile(trans_reg_scores, lower_percentile)
    trans_reg_upper = np.percentile(trans_reg_scores, upper_percentile)
    ground_truth_lower = np.percentile(ground_truth_scores, lower_percentile)
    ground_truth_upper = np.percentile(ground_truth_scores, upper_percentile)

    # Filter the data within the percentile range
    trans_reg_filtered = trans_reg_scores[(trans_reg_scores >= trans_reg_lower) & (trans_reg_scores <= trans_reg_upper)]
    ground_truth_filtered = ground_truth_scores[(ground_truth_scores >= ground_truth_lower) & (ground_truth_scores <= ground_truth_upper)]

    # Combine both into a single DataFrame for comparison
    scores_df = pd.DataFrame({
        'Score': np.concatenate([trans_reg_filtered, ground_truth_filtered]),
        'Source': ['True Negative'] * len(trans_reg_filtered) + ['True Positive (ground truth)'] * len(ground_truth_filtered)
    })

    # Plot the violin plot without outliers
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Source', y='Score', data=scores_df, inner="quartile")
    plt.ylabel('log2 Score')
    plt.xlabel('')

    if CELL_POP == True:
        plt.title(f'Cell Population TRP Scores (no outliers)')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/cell_pop_trans_reg_violin_plot_no_outliers.png', dpi=300)
    else:
        plt.title(f'{CELL_TYPE.capitalize()} TRP Scores (no outliers)')
        plt.tight_layout()
        plt.savefig(f'{RESULT_DIR}/{CELL_TYPE}/trans_reg_violin_plot_no_outliers.png', dpi=300)

    plt.close()
    logging.info("Violin plot without outliers saved.")


def create_network_graph(ground_truth_df: pd.DataFrame, output_file: str):
    """Create a directed transcription factor-target gene regulatory network using networkx and save it as a GEXF file."""
    edges = [(row['TF'], row['TG'], row['Score']) for index, row in ground_truth_df.iterrows()]

    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    nx.write_gexf(G, output_file)
    logging.info(f"\nNetwork graph saved to '{output_file}'.")


def summarize_ground_truth_and_trans_reg(
    original_ground_truth_df: pd.DataFrame, 
    original_trans_reg_network: pd.DataFrame,
    ground_truth_df: pd.DataFrame, 
    trans_reg_network: pd.DataFrame, 
    decimal_places: int = 2
):
    """Generate summary statistics for both ground truth scores and trans-regulatory network scores and display them."""
    
    # Define the summary file path
    summary_file_path = f"{RESULT_DIR}/{CELL_TYPE if not CELL_POP else 'cell_pop'}/summary_statistics.txt"
    
    # Helper function for logging and writing a message
    def log_and_write(file, message):
        logging.info(message)
        file.write(message + '\n')
    
    # Helper function to summarize network data
    def summarize_network(df, label, file, decimal_places=2, thresholds=None):
        num_tfs = len(set(df["TF"]))
        num_tgs = len(set(df["TG"]))
        num_edges = len(df['Score'])
        
        mean_score = round(df['Score'].mean(), decimal_places)
        median_score = round(df['Score'].median(), decimal_places)
        stdev_score = round(df['Score'].std(), decimal_places)

        log_and_write(file, f'----- {label} Scores -----')
        log_and_write(file, f'\tTFs: {num_tfs}, TGs: {num_tgs}, edges: {num_edges}')
        log_and_write(file, f'\tMean: {mean_score}, Median: {median_score}, Stdev: {stdev_score}')

        if thresholds:
            above_threshold = df[df['Score'] > thresholds['lower']].copy()
            below_threshold = df[df['Score'] < thresholds['lower']].copy()
            
            log_and_write(file, f'\tNum scores ABOVE 1st percentile: {above_threshold.shape[0]} '
                                 f'({round((above_threshold.shape[0] / num_edges) * 100, decimal_places)}%)')
            log_and_write(file, f'\tNum scores BELOW 1st percentile: {below_threshold.shape[0]} '
                                 f'({round((below_threshold.shape[0] / num_edges) * 100, decimal_places)}%)')


    # Calculate threshold information for ground truth scores
    gt_mean = ground_truth_df['Score'].mean()
    gt_stdev = ground_truth_df['Score'].std()
    thresholds = {
        'lower': round(ground_truth_df['Score'].quantile(0.05), decimal_places)
    }

    # Ensure result directory exists
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    
    with open(summary_file_path, 'w') as summary_file:
        log_and_write(summary_file, '----- Summary Statistics -----')
        log_and_write(summary_file, f'Total number of ground truth edges: {original_ground_truth_df.shape[0]}')

        # Percentage summaries for TFs and TGs in TRN
        tf_percent = round((len(set(ground_truth_df['TF'])) / len(set(original_ground_truth_df['Source']))) * 100, decimal_places)
        tg_percent = round((len(set(ground_truth_df['TG'])) / len(set(original_ground_truth_df['Target']))) * 100, decimal_places)
        edge_percent = round((len(ground_truth_df['Score']) / original_ground_truth_df.shape[0]) * 100, decimal_places)

        log_and_write(summary_file, f'{len(set(ground_truth_df["TF"]))}/{len(set(original_ground_truth_df["Source"]))} '
                                     f'ground truth TFs in TRN ({tf_percent}%)')
        log_and_write(summary_file, f'{len(set(ground_truth_df["TG"]))}/{len(set(original_ground_truth_df["Target"]))} '
                                     f'ground truth TGs in TRN ({tg_percent}%)')
        log_and_write(summary_file, f'{len(ground_truth_df["Score"])}/{original_ground_truth_df.shape[0]} '
                                     f'ground truth edges in TRN ({edge_percent}%)')
        
        # TRN and non-TRN edges
        trn_total_edges = original_trans_reg_network.shape[0] * original_trans_reg_network.shape[1]
        non_trn_edges = trn_total_edges - len(ground_truth_df['Score'])
        percent_in_trn = round((len(ground_truth_df['Score']) / trn_total_edges) * 100, decimal_places)

        log_and_write(summary_file, f'\nTotal number of TRN edges: {trn_total_edges}')
        log_and_write(summary_file, f'\tNumber of TRN edges NOT in ground truth network: {non_trn_edges}')
        log_and_write(summary_file, f'\tPercent of TRN represented by ground truth: {percent_in_trn}%\n')

        # Summarize both ground truth and trans-regulatory networks
        summarize_network(ground_truth_df, 'Ground truth TRN', summary_file, decimal_places, thresholds)
        summarize_network(trans_reg_network, 'Non-Ground Truth TRN', summary_file, decimal_places, thresholds)
        
def plot_auroc(ground_truth_df: pd.DataFrame, trans_reg_minus_ground_truth_df: pd.DataFrame):
    # Define the lower threshold
    lower_threshold = ground_truth_df['Score'].quantile(0.05)

    # Classify ground truth scores
    ground_truth_df['true_interaction'] = 1  # All entries in ground_truth_df are true interactions
    ground_truth_df['predicted_interaction'] = np.where(
        ground_truth_df['Score'] >= lower_threshold, 1, 0)  # 1 for TP, 0 for FN

    # Count ground truth classifications
    tp_count = ground_truth_df[ground_truth_df['predicted_interaction'] == 1].shape[0]
    fn_count = ground_truth_df[ground_truth_df['predicted_interaction'] == 0].shape[0]
    logging.info(f"\nGround Truth Scores: TP={tp_count}, FN={fn_count}")

    # Classify non-ground truth scores (trans_reg_minus_ground_truth_df)
    trans_reg_minus_ground_truth_df['true_interaction'] = 0  # All entries are non-interactions
    trans_reg_minus_ground_truth_df['predicted_interaction'] = np.where(
        trans_reg_minus_ground_truth_df['Score'] >= lower_threshold, 1, 0)  # 1 for FP, 0 for TN

    # Count non-ground truth classifications
    fp_count = trans_reg_minus_ground_truth_df[trans_reg_minus_ground_truth_df['predicted_interaction'] == 1].shape[0]
    tn_count = trans_reg_minus_ground_truth_df[trans_reg_minus_ground_truth_df['predicted_interaction'] == 0].shape[0]
    logging.info(f"Non-Ground Truth Scores: FP={fp_count}, TN={tn_count}")

    # Concatenate dataframes for AUC and further analysis
    auc_df = pd.concat([ground_truth_df, trans_reg_minus_ground_truth_df])

    # Calculate the confusion matrix to confirm the results programmatically
    from sklearn.metrics import confusion_matrix

    y_true = auc_df['true_interaction']
    y_pred = auc_df['predicted_interaction']
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    logging.info(f'\nTrue Positives: {tp}')
    logging.info(f'False Positives: {fp}')
    logging.info(f'True Negatives: {tn}')
    logging.info(f'False Negatives: {fn}')

    # Calculate ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, auc_df['Score'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Trans-Regulatory Potential Score Classification')
    plt.legend(loc="lower right")
    plt.grid()

    # Save the plot based on CELL_POP and CELL_TYPE
    save_path = f'{RESULT_DIR}/AUC.png' if CELL_POP else f'{RESULT_DIR}/{CELL_TYPE}/AUC.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    logging.info(f'Saved AUROC to {save_path}')

    plt.close()


def main():
    """Main function to run the analysis of the LINGER output"""

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # ----- DATA LOADING -----
    tf_re_binding, cis_reg_network, trans_reg_network = load_data()
    ground_truth = load_ground_truth()

    # ----- GENERAL INFORMATION -----
    logging.info(f'\n----- TF-RE Binding Potential -----')
    logging.info(f'REs (rows): {tf_re_binding.shape[0]}')
    logging.info(f'TFs (columns): {tf_re_binding.shape[1]}')
    logging.info(tf_re_binding.head())

    logging.info(f'\n----- Cis-Regulatory Network -----')
    logging.info(f'Network size: {cis_reg_network.shape[0]}')
    logging.info(cis_reg_network.head())

    logging.info(f'\n----- Trans-Regulatory Network -----')
    logging.info(f'Number of TFs (columns): {trans_reg_network.shape[1]}')
    logging.info(f'Number of TGs (rows): {trans_reg_network.shape[0]}')
    logging.info(trans_reg_network.head())

    # Process the TF-TG pairs and retrieve scores from the trans-regulatory network
    tf_list, tg_list, value_list, num_nan = process_tf_tg_pairs(ground_truth, trans_reg_network)

    print(f'TFs: {set(tf_list)}')

    # Save the processed ground truth scores
    ground_truth_df = save_ground_truth_scores(tf_list, tg_list, value_list)

    logging.info(f'\n----- Ground Truth Network -----')
    logging.info(f'Number of TF-TG pairs: {ground_truth.shape[0]}')
    logging.info(ground_truth.head())

    logging.info(f'\nGround truth scores')
    logging.info(f'Number of TF-TG pairs: {ground_truth_df.shape[0]}')
    logging.info(ground_truth_df.head())

    # ----- Creating a dataframe of non-ground truth TRP scores (True Negative) -----
    # Reset index so the TGs become a column
    trans_reg_network.reset_index(inplace=True)
    logging.info('\ntrans_reg_network')
    logging.info(trans_reg_network.head())

    # Melt the TRP dataset to have it in the same format as the ground truth
    # Rename the second column to TF
    trans_reg_net_melted = pd.melt(trans_reg_network, id_vars=['index'], var_name='TF', value_name='Score')
    logging.info('\ntrans_reg_net_melted')
    logging.info(trans_reg_net_melted.head())

    # Rename the 'index' column as 'TG'
    trans_reg_net_melted.rename(columns={'index': 'TG'}, inplace=True)

    # Step 1: Ensure that columns are properly formatted
    trans_reg_net_melted['TF'] = trans_reg_net_melted['TF'].astype(str).str.strip()
    trans_reg_net_melted['TG'] = trans_reg_net_melted['TG'].astype(str).str.strip()
    ground_truth_df['TF'] = ground_truth_df['TF'].astype(str).str.strip()
    ground_truth_df['TG'] = ground_truth_df['TG'].astype(str).str.strip()

    # Step 2: Perform the merge to identify non-ground-truth pairs
    difference_df = pd.merge(
        trans_reg_net_melted, 
        ground_truth_df[['TF', 'TG']], 
        on=['TF', 'TG'], 
        how='left', 
        indicator=True
        )

    logging.info(f'Difference df after merge:\n{difference_df.head()}')

    # Select only entries that are not in the ground truth
    trans_reg_minus_ground_truth_df = difference_df[difference_df['_merge'] == 'left_only'].copy()

    # Drop unnecessary columns and rename for consistency
    trans_reg_minus_ground_truth_df = (
        trans_reg_minus_ground_truth_df.drop(columns=['_merge'])
        .rename(columns={'Score_x': 'Score'})
    )
    
    # Ensure there are no overlaps by checking for any TF-TG pairs that appear in both dataframes
    overlap_check = pd.merge(
        trans_reg_minus_ground_truth_df[['TF', 'TG']], 
        ground_truth_df[['TF', 'TG']], 
        on=['TF', 'TG'], 
        how='inner'
    )

    if overlap_check.shape[0] > 0:
        logging.warning("There are still ground truth pairs in trans_reg_minus_ground_truth_df!")

    # Check for and drop any duplicates in the filtered dataframe
    trans_reg_minus_ground_truth_df = trans_reg_minus_ground_truth_df.drop_duplicates(subset=['TF', 'TG'])

    # Convert the scores to log2
    trans_reg_minus_ground_truth_df['Score'] = np.log2(trans_reg_minus_ground_truth_df['Score'])
    ground_truth_df['Score'] = np.log2(ground_truth_df['Score'])

    # Keep only the TFs and TGs that are shared between the datasets
    shared_tfs = set(trans_reg_minus_ground_truth_df['TF']).intersection(set(ground_truth_df['TF']))
    shared_tgs = set(trans_reg_minus_ground_truth_df['TG']).intersection(set(ground_truth_df['TG']))

    # Filter each dataframe to include only the shared TFs and TGs
    trans_reg_minus_ground_truth_df = trans_reg_minus_ground_truth_df[
        trans_reg_minus_ground_truth_df['TF'].isin(shared_tfs) & 
        trans_reg_minus_ground_truth_df['TG'].isin(shared_tgs)
    ]

    ground_truth_df = ground_truth_df[
        ground_truth_df['TF'].isin(shared_tfs) & 
        ground_truth_df['TG'].isin(shared_tgs)
    ]



    # Generating summary statistics for the Score column
    summarize_ground_truth_and_trans_reg(
        ground_truth,
        trans_reg_network,
        ground_truth_df,
        trans_reg_minus_ground_truth_df,
        decimal_places=2
    )

    plot_auroc(ground_truth_df, trans_reg_minus_ground_truth_df)


    # Find any TFs that are present in the ground truth dataset that are not in the e full dataset
    missing_tfs = [tf for tf in set(ground_truth_df["TF"]) if tf not in tf_list]

    if len(missing_tfs) > 0:
        logging.info(f'TFs present in ground truth missing in the trans-regulatory network')
        for i in missing_tfs:
            logging.info(i)

    # ----- FIGURES -----
    # Histogram distribution of trans-regulatory potential scores and ground truth scores
    plot_trans_reg_distribution(trans_reg_minus_ground_truth_df, ground_truth_df)

    plot_precision_recall_curve(ground_truth_df, trans_reg_minus_ground_truth_df)

    plot_trans_reg_distribution_with_thresholds(ground_truth_df, trans_reg_minus_ground_truth_df)

    # Plot a box and whisker plot of the trans-regulatory potential scores and ground truth scores
    plot_box_whisker(trans_reg_minus_ground_truth_df, ground_truth_df)

    # Plot a violin plot of the trans-regulatory potential scores and ground truth scores
    plot_violin(trans_reg_minus_ground_truth_df, ground_truth_df)

    # Violin plot with outliers removed
    plot_violin_without_outliers(trans_reg_minus_ground_truth_df, ground_truth_df)


    # ----- NETWORKX NETWORK -----
    # Create and save the network graph
    create_network_graph(ground_truth_df, f'{RESULT_DIR}/transcription_factor_target_gene_network.gexf')

if __name__ == '__main__':
    main()