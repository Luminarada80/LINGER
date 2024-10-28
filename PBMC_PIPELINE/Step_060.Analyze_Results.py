import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import logging
import os
import sys
import argparse
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

from linger import Benchmk
import PBMC_PIPELINE.shared_variables as shared_variables

# ----- THESE VARIABLES NEED TO CHANGE DEPENDING ON DATASET -----
CHIP_SEQ_GROUND_TRUTH_PATH: str = f'/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_PBMC_CISTROME/PBMC_cistrome_ground_truth.csv'
RESULT_DIR: str = '/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/PBMC_CISTROME_RESULTS'

CELL_TYPE_TF_DICT: dict = {
    'classical monocytes': {'CTCF', 'IRF1', 'RUNX1', 'SPI1', 'STAT1'},
    'naive B cells': {'CTCF', 'IRF4', 'MYC'},
    'naive CD4 T cells': {'ETS1', 'FOXP3', 'REST', 'RUNX1'},
    'myeloid DC': {'RUNX1'}
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
    ground_truth: pd.DataFrame = pd.read_csv(CHIP_SEQ_GROUND_TRUTH_PATH, sep=' ', header=None)
    return ground_truth


def process_tf_tg_pairs(ground_truth: pd.DataFrame, trans_reg_network: pd.DataFrame):
    """Process ground truth TF-TG pairs and retrieve the corresponding scores from the trans-regulatory network.
       Only considers TF-TG pairs for specified cell type TFs."""
    
    tf_list, tg_list, value_list = [], [], []
    num_nan: int = 0
    num_rows: int = 0

    for index, row in ground_truth.iterrows():
        tf, tg = row[0], row[1]
        
        # If looking at a cell type, only use TFs specific to that cell type
        # Check if the TF is in the cell type-specific TF list
        if CELL_POP == False:
            if tf in CELL_TYPE_TF_DICT[CELL_TYPE]:
                try:
                    # Get the trans-regulatory score for the TF-TG pair
                    value = trans_reg_network.loc[tg, tf]
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

    # Handle zeros and small values by adding a small constant (1e-6)
    linger_scores = np.where(linger_scores > 0, linger_scores, 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Plot histograms for the LINGER trans-reg network and ground truth network
    plt.hist(np.log2(linger_scores), bins=150, log=True, alpha=0.7, label='True negative (non-ground truth scores)')
    plt.hist(np.log2(ground_truth_scores), bins=150, log=True, alpha=0.7, label='True positive (ground truth scores)')

    
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


def plot_box_whisker(trans_reg_network_minus_ground_truth: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """Create box and whisker plots to compare trans-regulatory network scores and ground truth scores."""
    # Handle zeros and small values by adding a small constant (1e-6)
    trans_reg_scores = np.where(trans_reg_network_minus_ground_truth['Score'] > 0, trans_reg_network_minus_ground_truth['Score'], 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = np.log2(trans_reg_scores)
    ground_truth_scores = np.log2(ground_truth_scores)
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
    trans_reg_scores = np.where(trans_reg_network_minus_ground_truth['Score'] > 0, trans_reg_network_minus_ground_truth['Score'], 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = np.log2(trans_reg_scores)
    ground_truth_scores = np.log2(ground_truth_scores)

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
    trans_reg_scores = np.where(trans_reg_network_minus_ground_truth['Score'] > 0, trans_reg_network_minus_ground_truth['Score'], 1e-6)
    ground_truth_scores = np.where(ground_truth_df['Score'] > 0, ground_truth_df['Score'], 1e-6)

    # Apply log2 transformation after ensuring no zero or negative values
    trans_reg_scores = np.log2(trans_reg_scores)
    ground_truth_scores = np.log2(ground_truth_scores)

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


def summarize_ground_truth_and_trans_reg(ground_truth_df: pd.DataFrame, trans_reg_network: pd.DataFrame, decimal_places: int = 2):
    """Generate summary statistics for both ground truth scores and trans-regulatory network scores and display them."""
    
    # Generate summary statistics for the 'Score' column in the ground_truth_df
    ground_truth_stats = ground_truth_df['Score'].describe()
    
    # Generate summary statistics for the flattened trans_reg_network scores
    trans_reg_stats = pd.Series(trans_reg_network.values.flatten()).describe()

    # Combine the two sets of summary statistics into a single DataFrame
    summary_stats = pd.DataFrame({
        'Ground Truth': ground_truth_stats,
        'Trans-Regulatory Network': trans_reg_stats
    })

    # Round the summary statistics to the specified number of decimal places
    summary_stats = summary_stats.round(decimal_places)

    # Print the rounded summary statistics to the terminal
    print("\nSummary Statistics for Ground Truth Scores and Trans-Regulatory Network Scores (Rounded):")
    print(summary_stats)

    # Save the summary to a csv file
    if CELL_POP == True:
        summary_stats.to_csv(f'{RESULT_DIR}/cell_pop_summary_statistics.csv')
    else:
        summary_stats.to_csv(f'{RESULT_DIR}/{CELL_TYPE}/summary_statistics.csv')

    logging.info("Summary statistics saved to 'summary_statistics.csv'.")


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
    logging.info(ground_truth_df.head())

    # ----- Creating a dataframe of non-ground truth TRP scores (True Negative) -----
    # Reset index so the TGs become a column
    trans_reg_network.reset_index(inplace=True)

    # Melt the TRP dataset to have it in the same format as the ground truth
    trans_reg_net_melted = pd.melt(trans_reg_network, id_vars=['index'], var_name='TF', value_name='Score')

    # Add the column header for TGs
    trans_reg_net_melted.rename(columns={'index': 'TG'}, inplace=True)

    # Perform a left merge to find rows in trans_reg_pairs that are not in ground_truth_pairs
    difference_df = pd.merge(trans_reg_net_melted, ground_truth_df, on=['TF', 'TG'], how='left', indicator=True)

    # Store the true negatives as the TRP pairs not in the ground truth network (pairs that don't have a score in the ground truth dataframe)
    trans_reg_minus_ground_truth_df = difference_df[difference_df['_merge'] == 'left_only']

    # Keep only the relevant columns and rename for consistency
    trans_reg_minus_ground_truth_df = trans_reg_minus_ground_truth_df.drop(columns=['_merge', 'Score_y']).rename(columns={'Score_x': 'Score'})

    logging.info(f'\n----- Summary Statistics -----')
    # Generating summary statistics for the Score column
    summarize_ground_truth_and_trans_reg(ground_truth_df, trans_reg_minus_ground_truth_df, decimal_places=2)

    # Calculate metrics to see how many of the ground truth TFs, TGs, and pairs are in the full 
    # trans-regulatory potential dataset
    num_rows = ground_truth.shape[0]
    
    num_tf_ground_truth = len(set(ground_truth_df["TF"]))
    num_tf_trans_net = len(set(tf_list))

    num_tg_ground_truth = len(set(ground_truth_df['TG']))
    tg_overlap = len([tg for tg in set(ground_truth_df["TG"]) if tg not in tf_list])

    logging.info(f'\n{num_tf_trans_net}/{num_tf_ground_truth} of ground truth TFs are represented in the trans-regulatory network')
    logging.info(f'{tg_overlap}/{num_tg_ground_truth} of ground truth TGs are represented in the trans-regulatory network')

    logging.info(f'{num_nan} / {num_rows} ({round(num_nan/num_rows*100,2)}%) of TF to TG pairs did not have a score in the LINGER dataset')

    # Find any TFs that are present in the ground truth dataset that are not in the e full dataset
    missing_tfs = [tf for tf in set(ground_truth_df["TF"]) if tf not in tf_list]

    if len(missing_tfs) > 0:
        logging.info(f'TFs present in ground truth missing in the trans-regulatory network')
        for i in missing_tfs:
            logging.info(i)

    # ----- FIGURES -----
    # Histogram distribution of trans-regulatory potential scores and ground truth scores
    plot_trans_reg_distribution(trans_reg_minus_ground_truth_df, ground_truth_df)

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