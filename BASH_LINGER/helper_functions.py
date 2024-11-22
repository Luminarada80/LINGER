import logging
import pandas as pd
import math
import scanpy as sc
from typing import TextIO
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')   

# Set font to Arial and adjust font sizes
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  # General font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label size
    'ytick.labelsize': 14,  # Y-axis tick label size
    'legend.fontsize': 14  # Legend font size
})

def log_and_write(file: TextIO, message: str) -> None:
    """
    Helper function for logging and writing a message to a file.
    
    Parameters:
        file (TextIO): An open file object where the message will be written.
        message (str): The message to be logged and written to the file.
    """
    logging.info(message)
    file.write(message + '\n')

def source_target_cols_uppercase(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the Source and Target columns to uppercase for a dataframe
    
    Args:
        df (pd.DataFrame): GRN dataframe with "Source" and "Target" gene name columns

    Returns:
        pd.DataFrame: The same dataframe with uppercase gene names
    """
    df["Source"] = df["Source"].str.upper()
    df["Target"] = df["Target"].str.upper()

def create_standard_dataframe(
    inferred_network_df: pd.DataFrame,
    source_col: str = None,
    target_col: str = None,
    score_col: str = None) -> pd.DataFrame:
    
    """
    Standardizes inferred GRN dataframes to have three columns with "Source", "Target", and "Score".
    Makes all TF and TG names uppercase.
    
    Parameters:
        inferred_network_df (pd.dataframe):
            Inferred GRN dataframe
        
        source_col (str):
            The column name that should be used for the TFs. Default is "Source"
        
        target_col (str):
            The column name that should be used for the TGs. Default is "Target"
        
        score_col (str):
            The column name that should be used as the score. Default is "Score"
    
    Returns:
        standardized_df (pd.DataFrame):
            A dataframe with three columns: "Source", "Target, and "Score"
    """

    source_col = source_col.capitalize() if source_col else "Source"
    target_col = target_col.capitalize() if target_col else "Target"
    score_col = score_col.capitalize() if score_col else "Score"
    
    # Capitalize the column names for consistency
    inferred_network_df.columns = inferred_network_df.columns.str.capitalize()
        
    
    # Detect if the DataFrame needs to be melted
    if "Source" in inferred_network_df.columns and "Target" in inferred_network_df.columns:
        # print("\nDataFrame appears to be in long format; no melting is required.")
                        
        # If no melting is required, we just rename columns directly
        melted_df = inferred_network_df.rename(columns={source_col: "Source", target_col: "Target", score_col: "Score"})
        
        logging.info(f'\t{len(set(melted_df["Source"])):,} TFs, {len(set(melted_df["Target"])):,} TGs, and {len(melted_df["Score"]):,} edges')

    
    # The dataframe needs to be melted, there are more than 3 columns and no "Source" or "Target" columns
    elif inferred_network_df.shape[1] > 3:
        
        num_rows, num_cols = inferred_network_df.shape
        
        logging.debug(f'Original dataframe has {num_rows} rows and {num_cols} columns')
        
        logging.debug(f'\nOld df before melting:')
        logging.debug(inferred_network_df.head())
        
        # TFs are columns, TGs are rows
        if num_rows >= num_cols:
            logging.info(f'\t{num_cols} TFs, {num_rows} TGs, and {num_cols * num_rows} edges')
            # Transpose the columns and rows to prepare for melting
            inferred_network_df = inferred_network_df.T
            
            # Reset the index to make the TFs a column named 'Source'
            inferred_network_df = inferred_network_df.reset_index()
            inferred_network_df = inferred_network_df.rename(columns={'index': 'Source'})  # Rename the index column to 'Source'
    
            melted_df = inferred_network_df.melt(id_vars="Source", var_name="Target", value_name="Score")
            
        # TFs are rows, TGs are columns
        elif num_cols > num_rows:
            logging.info(f'\t{num_rows} TFs, {num_cols} TGs, and {num_cols * num_rows} edges')
            
            # Reset the index to make the TFs a column named 'Source'
            inferred_network_df = inferred_network_df.reset_index()
            inferred_network_df = inferred_network_df.rename(columns={'index': 'Source'})  # Rename the index column to 'Source'
    
            melted_df = inferred_network_df.melt(id_vars="Source", var_name="Target", value_name="Score")

    # Capitalize and strip whitespace for consistency
    source_target_cols_uppercase(melted_df)

    # Select and order columns as per new standard
    standardized_df = melted_df[["Source", "Target", "Score"]]
    
    logging.debug(f'\nNew df after standardizing:')
    logging.debug(standardized_df.head())
    
    return standardized_df

def add_inferred_scores_to_ground_truth(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame,
    score_column_name: str = "Score"
    ) -> pd.DataFrame:
    
    """
    Merges the inferred network scores with the ground truth dataframe, returns a ground truth network
    with a column containing the inferred Source -> Target score for each row.
    
    Parameters:
        ground_truth_df (pd.DataFrame):
            The ground truth dataframe. Columns should be "Source" and "Target"
            
        inferred_network_df (pd.DataFrame):
            The inferred GRN dataframe. Columns should be "Source", "Target", and "Score"
        
        score_column_name (str):
            Renames the "Score" column to a specific name, used if multiple datasets are being compared.
    
    """
    # Make sure that ground truth and inferred network "Source" and "Target" columns are uppercase
    source_target_cols_uppercase(ground_truth_df)
    source_target_cols_uppercase(inferred_network_df)
    
    # Take the Source and Target from the ground truth and the Score from the inferred network to create a new df
    ground_truth_with_scores = pd.merge(
        ground_truth_df, 
        inferred_network_df[["Source", "Target", "Score"]], 
        left_on=["Source", "Target"], 
        right_on=["Source", "Target"], 
        how="left"
    ).rename(columns={"Score": score_column_name})

    
    return ground_truth_with_scores

def remove_ground_truth_edges_from_inferred(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Removes ground truth edges from the inferred network after setting the ground truth scores.
    
    After this step, the inferred network does not contain any ground truth edge scores. This way, the 
    inferred network and ground truth network scores can be compared.
    
    Parameters:
        ground_truth_df (pd.DataFrame):
            Ground truth df with columns "Source" and "Target" corresponding to TFs and TGs
        
        inferred_network_df (pd.DataFrame):
            The inferred GRN df with columns "Source" and "Target" corresponding to TFs and TGs

    Returns:
        inferred_network_no_ground_truth (pd.DataFrame):
            The inferred GRN without the ground truth scores
    """
    # Make sure that ground truth and inferred network "Source" and "Target" columns are uppercase
    source_target_cols_uppercase(ground_truth_df)
    source_target_cols_uppercase(inferred_network_df)
    
    # Get a list of the ground truth edges to separate 
    ground_truth_edges = set(zip(ground_truth_df['Source'], ground_truth_df['Target']))
    
    # Create a new dataframe without the ground truth edges
    inferred_network_no_ground_truth = inferred_network_df[
        ~inferred_network_df.apply(lambda row: (row['Source'], row['Target']) in ground_truth_edges, axis=1)
    ]
    
    return inferred_network_no_ground_truth

def remove_tf_tg_not_in_ground_truth(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame
    ) -> pd.DataFrame:
    
    """
    Only keeps inferred network TFs and TGs that are also in the ground truth network.
    
    Parameters:
        ground_truth_df (pd.DataFrame):
            Ground truth df with columns "Source" and "Target" corresponding to TFs and TGs
        
        inferred_network_df (pd.DataFrame):
            The inferred GRN df with columns "Source" and "Target" corresponding to TFs and TGs
    
    """
    # Make sure that ground truth and inferred network "Source" and "Target" columns are uppercase
    source_target_cols_uppercase(ground_truth_df)
    source_target_cols_uppercase(inferred_network_df)
    
    # Extract unique TFs and TGs from the ground truth network
    ground_truth_tfs = set(ground_truth_df['Source'])
    ground_truth_tgs = set(ground_truth_df['Target'])
    
    # Subset cell_oracle_network to contain only rows with TFs and TGs in the ground_truth
    aligned_inferred_network = inferred_network_df[
        (inferred_network_df['Source'].isin(ground_truth_tfs)) &
        (inferred_network_df['Target'].isin(ground_truth_tgs))
    ]
    
    return aligned_inferred_network

def calculate_accuracy_metrics(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame,
    lower_threshold: int = None,
    num_edges_for_early_precision: int = 1000,
    ):
    
    """
    Calculates accuracy metrics for an inferred network.
    
    Uses a lower threshold as the cutoff between true and false values. The 
    default lower threshold is set as 1 standard deviation below the mean 
    ground truth scores.
    
    True Positive: ground truth scores above the lower threshold
    False Positive: non-ground truth scores above the lower threshold
    True Negative: non-ground truth scores below the lower threshold
    False Negative: ground truth scores below the lower threshold
    
    Parameters:
        ground_truth_df (pd.DataFrame):
            The ground truth dataframe with inferred network scores
            
        inferred_network_df (pd.DataFrame):
            The inferred GRN dataframe

    Returns:
        summary_dict (dict):
            A dictionary of the TP, TN, FP, and FN scores along with y_true and y_pred
            
        accuracy_metrics (dict):
            A dictionary of the accuracy metrics and their values

    """
    
    if not lower_threshold:
        lower_threshold = np.mean(ground_truth_df['Score']) - np.std(ground_truth_df['Score']) 
    
    # Classify ground truth scores
    ground_truth_df['true_interaction'] = 1
    ground_truth_df['predicted_interaction'] = np.where(
        ground_truth_df['Score'] >= lower_threshold, 1, 0)

    # Classify non-ground truth scores (trans_reg_minus_ground_truth_df)
    inferred_network_df['true_interaction'] = 0
    inferred_network_df['predicted_interaction'] = np.where(
        inferred_network_df['Score'] >= lower_threshold, 1, 0)

    # Concatenate dataframes for AUC and further analysis
    auc_df = pd.concat([ground_truth_df, inferred_network_df])
    print(auc_df)

    # Calculate the confusion matrix
    y_true = auc_df['true_interaction']
    y_pred = auc_df['predicted_interaction']
    y_scores = auc_df['Score'].dropna()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculations for accuracy metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    jaccard_index = tp / (tp + fp + fn)

    # Weighted Jaccard Index
    weighted_tp = ground_truth_df.loc[ground_truth_df['predicted_interaction'] == 1, 'Score'].sum()
    weighted_fp = inferred_network_df.loc[inferred_network_df['predicted_interaction'] == 1, 'Score'].sum()
    weighted_fn = ground_truth_df.loc[ground_truth_df['predicted_interaction'] == 0, 'Score'].sum()
    weighted_jaccard_index = weighted_tp / (weighted_tp + weighted_fp + weighted_fn)
    
    # Early Precision Rate for top 1000 predictions
    top_edges = auc_df.nlargest(int(num_edges_for_early_precision), 'Score')
    early_tp = top_edges[top_edges['true_interaction'] == 1].shape[0]
    early_fp = top_edges[top_edges['true_interaction'] == 0].shape[0]
    early_precision_rate = early_tp / (early_tp + early_fp)
    
    accuracy_metrics = {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
        }
    
    summary_dict = {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "jaccard_index": jaccard_index,
        "weighted_jaccard_index": weighted_jaccard_index,
        "early_precision_rate": early_precision_rate,
    }
    
    return summary_dict, accuracy_metrics

def plot_multiple_histogram_with_thresholds(ground_truth_dict: dict, inferred_network_dict: dict, result_dir: str) -> None:
    """
    Generates histograms of the TP, FP, TN, FN score distributions for each method. Uses a lower threshold of 1 stdev below
    the mean ground truth score. 

    Args:
        ground_truth_dict (dict): _description_
        inferred_network_dict (dict): _description_
        result_dir (str): _description_
    """
    
    num_methods = len(ground_truth_dict.keys())

    # Maximum columns per row
    max_cols = 4

    # Calculate the number of rows and columns
    num_cols = min(num_methods, max_cols)  # Up to 4 columns
    num_rows = math.ceil(num_methods / max_cols)  # Rows needed to fit all methods

    print(f"Number of rows: {num_rows}, Number of columns: {num_cols}")
    
    plt.figure(figsize=(18, 8))

    # Plot for each method
    for i, method_name in enumerate(ground_truth_dict.keys()):  
        
        # Extract data for the current method
        ground_truth_scores = ground_truth_dict[method_name]['Score'].dropna()
        inferred_scores = inferred_network_dict[method_name]['Score'].dropna()
        
        plt.subplot(num_rows, num_cols, i+1)
        
        # Define the threshold
        lower_threshold = np.mean(ground_truth_scores) - np.std(ground_truth_scores)

        # Define consistent bin edges for the entire dataset
        num_bins = 300
        bin_edges = np.histogram_bin_edges(
            np.concatenate([ground_truth_scores, inferred_scores]), bins=num_bins
        )

        # Split data into below and above threshold
        tp = ground_truth_scores[ground_truth_scores > lower_threshold]
        fn = ground_truth_scores[ground_truth_scores <= lower_threshold]
        
        fp = inferred_scores[inferred_scores >= lower_threshold]
        tn = inferred_scores[inferred_scores < lower_threshold]
        
        # Debugging: Print counts for each category
        print(f"TP count: {len(tp)}, TN count: {len(tn)}, FP count: {len(fp)}, FN count: {len(fn)}")
        
        # Plot histograms for Oracle Score categories with consistent bin sizes
        plt.hist(tn, bins=bin_edges, alpha=1, color='#b6cde0', label='True Negative (TN)')
        plt.hist(fp, bins=bin_edges, alpha=1, color='#4195df', label='False Positive (FP)')
        plt.hist(tp, bins=bin_edges, alpha=1, color='#dc8634', label='True Positive (TP)')
        plt.hist(fn, bins=bin_edges, alpha=1, color='#efc69f', label='False Negative (FN)')

        # Plot Oracle threshold line
        plt.axvline(x=lower_threshold, color='black', linestyle='--', linewidth=2)
        plt.title(f"{method_name} Score Distribution")
        plt.xlabel(f"log2 {method_name} Score")
        plt.xlim([-20,20])
        plt.ylabel("Frequency")
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{result_dir}/Histogram_Multiple_Method_GRN_Scores.png')

def find_inferred_network_accuracy_metrics(ground_truth: pd.DataFrame, inferred_network: pd.DataFrame, lower_threshold = None):

    # Set the ground truth and inferred network scores to log2 scale
    inferred_network["Score"] = np.log2(inferred_network["Score"])
    ground_truth["Score"] = np.log2(ground_truth["Score"])
    
    # Set the default threshold as 1 stdev below the mean ground truth score
    if lower_threshold == None:
        mean = ground_truth["Score"].mean()
        std = ground_truth["Score"].std()
        
        lower_threshold = mean - 1 * std
    
    true_positive_df: pd.DataFrame = ground_truth[ground_truth["Score"] >= lower_threshold]["Score"]
    false_negative_df: pd.DataFrame = ground_truth[ground_truth["Score"] < lower_threshold]["Score"]
    
    false_positive_df: pd.DataFrame = inferred_network[inferred_network["Score"] >= lower_threshold]["Score"]
    true_negative_df: pd.DataFrame = inferred_network[inferred_network["Score"] < lower_threshold]["Score"]
    
    return true_positive_df, false_positive_df, true_negative_df, false_negative_df

def plot_histogram_with_threshold(
    dataset_label: str,
    true_negative_df: pd.DataFrame,
    false_positive_df: pd.DataFrame,
    false_negative_df: pd.DataFrame,
    true_positive_df: pd.DataFrame,
    lower_threshold: int | float = None
    ) -> plt.Figure:
    
    # Create the figure and axes
    fig, ax = plt.subplots()

    # Plot histograms for Oracle Score categories with calculated bin sizes
    ax.hist(true_negative_df.squeeze(), bins=75, alpha=1, color='#b6cde0', label='True Negative (TN)')
    ax.hist(false_positive_df.squeeze(), bins=150, alpha=1, color='#4195df', label='False Positive (FP)')
    ax.hist(false_negative_df.squeeze(), bins=75, alpha=1, color='#efc69f', label='False Negative (FN)')
    ax.hist(true_positive_df.squeeze(), bins=150, alpha=1, color='#dc8634', label='True Positive (TP)')

    # Plot Oracle threshold line
    ax.axvline(x=lower_threshold, color='black', linestyle='--', linewidth=2)
    ax.set_title(f"{dataset_label} Score Distribution")
    ax.set_xlabel(f"log2 {dataset_label} Score")
    ax.set_ylabel("Frequency")
    ax.legend()

    return fig

def plot_cell_expression_histogram(
    adata_rna: sc.AnnData, 
    output_dir: str, 
    cell_type: str = None,
    ymin: int | float = 0,
    ymax: int | float = 100,
    xmin: int | float = 0,
    xmax: int | float = None,
    filename: str = 'avg_gene_expr_hist',
    filetype: str = 'png'
    
    ):
    """
    Plots a histogram showing the distribution of cell gene expression by percent of the population.
    
    Assumes the data has cells as columns and genes as rows

    Args:
        adata_rna (sc.AnnData):
            scRNAseq dataset with cells as columns and genes as rows
        output_dir (str):
            Directory to save the graph in
        cell_type (str):
            Can specify the cell type if the dataset is a single cell type
        ymin (int | float):
            The minimum y-axis value (in percentage, default = 0)
        ymax (int | float):
            The maximum y-axis value (in percentage, default = 100)
        xmin (int | float):
            The minimum x-axis value (default = 0)
        xmax (int | float):
            The maximum x-axis value
        filename (str):
            The name of the file to save the figure as
        filetype (str):
            The file extension of the figure (default = png)
    """        
    
    n_cells = adata_rna.shape[0]
    n_genes = adata_rna.shape[1]
    
    # Create the histogram and calculate bin heights
    plt.hist(adata_rna.obs["n_genes"], bins=30, edgecolor='black', weights=np.ones_like(adata_rna.obs["n_genes"]) / n_cells * 100)
    
    if cell_type == None:
        plt.title(f'Distribution of the number of genes expressed by cells in the dataset')
    else: 
        plt.title(f'Distribution of the number of genes expressed by {cell_type}s in the dataset')
        
    plt.xlabel(f'Number of genes expressed ({n_genes} total genes)')
    plt.ylabel(f'Percentage of cells ({n_cells} total cells)')
    
    plt.yticks(np.arange(0, min(ymax, 100) + 1, 5), [f'{i}%' for i in range(0, min(ymax, 100) + 1, 5)])
    plt.ylabel(f'Percentage of cells ({n_cells} total cells)')
    
    plt.savefig(f'{output_dir}/{filename}.{filetype}', dpi=300)
    plt.close()