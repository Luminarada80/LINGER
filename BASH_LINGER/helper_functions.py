import logging
import pandas as pd
import math
import scanpy as sc
from typing import TextIO
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import re
import os

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
    
    Parameters
    ----------
        df (pd.DataFrame): GRN dataframe with "Source" and "Target" gene name columns

    Returns
    ----------
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
    
    Parameters
    ----------
        inferred_network_df (pd.dataframe):
            Inferred GRN dataframe
        
        source_col (str):
            The column name that should be used for the TFs. Default is "Source"
        
        target_col (str):
            The column name that should be used for the TGs. Default is "Target"
        
        score_col (str):
            The column name that should be used as the score. Default is "Score"
    
    Returns
    ----------
        standardized_df (pd.DataFrame):
            A dataframe with three columns: "Source", "Target, and "Score"
    """
    
    # Check to make sure the dataframe is input correctly
    if inferred_network_df.empty:
        raise ValueError("The input dataframe is empty. Please provide a valid dataframe.")

    source_col = source_col.capitalize() if source_col else "Source"
    target_col = target_col.capitalize() if target_col else "Target"
    score_col = score_col.capitalize() if score_col else "Score"
    
    # Capitalize the column names for consistency
    inferred_network_df.columns = inferred_network_df.columns.str.capitalize()
        
    # Detect if the DataFrame needs to be melted
    if "Source" in inferred_network_df.columns and "Target" in inferred_network_df.columns:  
        # Make sure that the gene names are all strings and dont include whitespace
        inferred_network_df["Source"] = inferred_network_df["Source"].astype(str).str.strip()
        inferred_network_df["Target"] = inferred_network_df["Target"].astype(str).str.strip()
                              
        # If no melting is required, we just rename columns directly
        melted_df = inferred_network_df.rename(columns={source_col: "Source", target_col: "Target", score_col: "Score"})
        
        logging.info(f'{len(set(melted_df["Source"])):,} TFs, {len(set(melted_df["Target"])):,} TGs, and {len(melted_df["Score"]):,} edges')

    
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
    
    # # Remove any infinite and NaN values from the Score column
    # standardized_df = standardized_df.replace([np.inf, -np.inf], np.nan)
    # standardized_df = standardized_df.dropna(subset=["Score"])
    
    logging.debug(f'\nNew df after standardizing:')
    logging.debug(standardized_df.head())
    
    # Validates the structure of the dataframe before returning it
    assert all(col in standardized_df.columns for col in ["Source", "Target", "Score"]), \
    "Standardized dataframe does not contain the required columns."
    
    return standardized_df

def add_inferred_scores_to_ground_truth(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame,
    score_column_name: str = "Score"
    ) -> pd.DataFrame:
    
    """
    Merges the inferred network Score with the ground truth dataframe, returns a ground truth network
    with a column containing the inferred Source -> Target score for each row.
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            The ground truth dataframe. Columns should be "Source" and "Target"
            
        inferred_network_df (pd.DataFrame):
            The inferred GRN dataframe. Columns should be "Source", "Target", and "Score"
        
        score_column_name (str):
            Renames the "Score" column to a specific name, used if multiple datasets are being compared.
    
    Returns
    ----------
        ground_truth_with_scores (pd.DataFrame):
            Ground truth dataframe with a "Score" column corresonding to the edge Score for each 
            Target Source pair from the inferred network
    
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
    Removes ground truth edges from the inferred network after setting the ground truth Score.
    
    After this step, the inferred network does not contain any ground truth edge Score. This way, the 
    inferred network and ground truth network Score can be compared.
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            Ground truth df with columns "Source" and "Target" corresponding to TFs and TGs
        
        inferred_network_df (pd.DataFrame):
            The inferred GRN df with columns "Source" and "Target" corresponding to TFs and TGs

    Returns
    ----------
        inferred_network_no_ground_truth (pd.DataFrame):
            The inferred GRN without the ground truth Score
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
    
    # Ensure there are no overlaps by checking for any TF-TG pairs that appear in both dataframes
    overlap_check = pd.merge(
        inferred_network_no_ground_truth[['Source', 'Target']], 
        ground_truth_df[['Source', 'Target']], 
        on=['Source', 'Target'], 
        how='inner'
    )

    # Check to make sure there is no overlap between the ground truth and inferred networks
    if overlap_check.shape[0] > 0:
        logging.warning("There are still ground truth pairs in trans_reg_minus_ground_truth_df!")
    
    return inferred_network_no_ground_truth

def remove_tf_tg_not_in_ground_truth(
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame
    ) -> pd.DataFrame:
    
    """
    Removes edges from the inferred network whether either the "Source" or "Target" gene is not
    found in the ground truth network.
    
    This allows for the evaluation of the GRN inference method by removing TFs and TGs that could
    not be found in the ground truth. The inferred edges may exist, but they will be evaluated as 
    incorrect if they are not seen in the ground truth network. This way, we can determine if an 
    inferred edge is true or false by its presence in the ground truth network and inferred edge score.
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            Ground truth df with columns "Source" and "Target" corresponding to TFs and TGs
        
        inferred_network_df (pd.DataFrame):
            The inferred GRN df with columns "Source" and "Target" corresponding to TFs and TGs
    
    Returns
    ----------
        aligned_inferred_network (pd.DataFrame):
            Returns the inferred network containing only rows with only edges where both the "Source"
            and "Target" genes are found in the ground truth network.
    
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
    ) -> tuple[dict, dict]:
    
    """
    Calculates accuracy metrics for an inferred network.
    
    Uses a lower threshold as the cutoff between true and false values. The 
    default lower threshold is set as 1 standard deviation below the mean 
    ground truth Score.
    
    True Positive: ground truth Score above the lower threshold
    False Positive: non-ground truth Score above the lower threshold
    True Negative: non-ground truth Score below the lower threshold
    False Negative: ground truth Score below the lower threshold
    
    Parameters
    ----------
        ground_truth_df (pd.DataFrame):
            The ground truth dataframe with inferred network Score
            
        inferred_network_df (pd.DataFrame):
            The inferred GRN dataframe

    Returns
    ----------
        summary_dict (dict):
            A dictionary of the TP, TN, FP, and FN Score along with y_true and y_pred.
                
        confusion_matrix_score_dict (dict):
            A dictionary of the accuracy metrics and their values

    """
        
    if lower_threshold == None:
        gt_mean = ground_truth_df['Score'].mean()
        gt_std = ground_truth_df['Score'].std()

        # Define the lower threshold
        lower_threshold = gt_mean - 1 * gt_std
    
    # Classify ground truth Score
    ground_truth_df['true_interaction'] = 1
    ground_truth_df['predicted_interaction'] = np.where(
        ground_truth_df['Score'] >= lower_threshold, 1, 0)

    # Classify non-ground truth Score (trans_reg_minus_ground_truth_df)
    inferred_network_df['true_interaction'] = 0
    inferred_network_df['predicted_interaction'] = np.where(
        inferred_network_df['Score'] >= lower_threshold, 1, 0)

    # Concatenate dataframes for AUC and further analysis
    auc_df = pd.concat([ground_truth_df, inferred_network_df])

    # Calculate the confusion matrix
    y_true = auc_df['true_interaction'].dropna()
    y_pred = auc_df['predicted_interaction'].dropna()
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
    
    accuracy_metric_dict = {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "jaccard_index": jaccard_index,
        "weighted_jaccard_index": weighted_jaccard_index,
        "early_precision_rate": early_precision_rate,
    }
    
    confusion_matrix_score_dict = {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
        }
    
    return accuracy_metric_dict, confusion_matrix_score_dict

def plot_multiple_histogram_with_thresholds(
    ground_truth_dict: dict,
    inferred_network_dict: dict,
    save_path: str
    ) -> None:
    """
    Generates histograms of the TP, FP, TN, FN score distributions for each method. Uses a lower threshold of 1 stdev below
    the mean ground truth score. 

    Parameters
    ----------
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

    # print(f"Number of rows: {num_rows}, Number of columns: {num_cols}")
    
    plt.figure(figsize=(18, 8))

    # Plot for each method
    for i, method_name in enumerate(ground_truth_dict.keys()):  
        
        # Extract data for the current method
        ground_truth_scores = ground_truth_dict[method_name]['Score'].dropna()
        inferred_scores = inferred_network_dict[method_name]['Score'].dropna()
        
        plt.subplot(num_rows, num_cols, i+1)
        
        # Define the threshold
        lower_threshold = np.mean(ground_truth_scores) - np.std(ground_truth_scores)
        
        mean = np.mean(np.concatenate([ground_truth_scores, inferred_scores]))
        std_dev = np.std(np.concatenate([ground_truth_scores, inferred_scores]))

        xmin = mean - 4 * std_dev
        xmax = mean + 4 * std_dev
        
        # Split data into below and above threshold
        tp = ground_truth_scores[(ground_truth_scores >= lower_threshold) & (ground_truth_scores < xmax)]
        fn = ground_truth_scores[(ground_truth_scores <= lower_threshold) & (ground_truth_scores > xmin)]
        
        fp = inferred_scores[(inferred_scores >= lower_threshold) & (inferred_scores < xmax)]
        tn = inferred_scores[(inferred_scores <= lower_threshold) & (inferred_scores > xmin)]
        
        # Define consistent bin edges for the entire dataset based on the number of values
        num_bins = 200
        all_scores = np.concatenate([tp, fn, fp, tn])
        bin_edges = np.linspace(np.min(all_scores), np.max(all_scores), num_bins)
        # bin_edges = np.sort(np.unique(np.append(bin_edges, lower_threshold)))
        
        # Plot histograms for Oracle Score categories with consistent bin sizes
        plt.hist(tn, bins=bin_edges, alpha=1, color='#b6cde0', label='True Negative (TN)')
        plt.hist(fn, bins=bin_edges, alpha=1, color='#efc69f', label='False Negative (FN)')
        
        # Plot the positive values on top to make sure there is no ovelap
        plt.hist(fp, bins=bin_edges, alpha=1, color='#4195df', label='False Positive (FP)')
        plt.hist(tp, bins=bin_edges, alpha=1, color='#dc8634', label='True Positive (TP)')

        # Plot threshold line
        plt.axvline(x=lower_threshold, color='black', linestyle='--', linewidth=2)
        plt.title(f"{method_name.capitalize()} Score Distribution")
        plt.xlabel(f"log2 {method_name.capitalize()} Score")
        plt.ylim(1, None)
        plt.xlim([-20,20])
        plt.ylabel("Frequency")
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=200)
    plt.close()

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

    Parameters
    ----------
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
    
def create_randomized_inference_scores(    
    ground_truth_df: pd.DataFrame,
    inferred_network_df: pd.DataFrame,
    lower_threshold: int = None,
    num_edges_for_early_precision: int = 1000,
    histogram_save_path: str = None
    ):
    
    """
    Uses random permutation to randomize the edge Score for inferred network and ground truth network dataframes.
    Randomly reshuffles and calculates accuracy metrics from the new Score.

    Parameters
    ----------
        ground_truth_df (pd.DataFrame): 
            The ground truth network dataframe containing columns "Source" and "Target"
        inferred_network_df (pd.DataFrame): 
            The inferred network dataframe containing columns "Source", "Target", and "Score"
        lower_threshold (int, optional): 
            Can optionally provide a lower threshold value. Defaults to 1 stdev below the ground truth mean.
        num_edges_for_early_precision (int, optional): 
            Can optionally provide the number of edges for calculating early precision rate. Defaults to 1000.
        histogram_save_path (str, optional):
            Can optionlly provide a save path to create a histogram of the randomized ground truth GRN scores
            vs. the randomized inferred network scores

    Returns
    ----------
        accuracy_metric_dict (dict):
            A dictionary containing the calculated accuracy metrics. Contains keys:
            "precision"
            "recall"
            "specificity"
            "accuracy"
            "f1_score"
            "jaccard_index"
            "weighted_jaccard_index"
            "early_precision_rate"
        
        confusion_matrix_dict (dict):
            A dictionary containing the results of the confution matrix and metrics for AUROC
            and AUPRC. Contains keys:
            "true_positive"
            "true_negative"
            "false_positive"
            "false_negative"
            "y_true"
            "y_pred"
            "y_scores"
    """
    
    # Create a copy of the inferred network and ground truth dataframe
    inferred_network_df_copy = inferred_network_df.copy()
    ground_truth_df_copy = ground_truth_df.copy()
    
    # Extract the Score for each edge
    inferred_network_score = inferred_network_df_copy["Score"]
    ground_truth_score = ground_truth_df_copy["Score"]
    
    # Combine the scores from the ground truth and inferred network
    total_scores = pd.concat(
        [ground_truth_df["Score"], inferred_network_df["Score"]]
    ).values
    
    # Randomly reassign scores back to the ground truth and inferred network
    resampled_inferred_network_scores = np.random.choice(total_scores, size=len(inferred_network_score), replace=True)
    resampled_ground_truth_scores = np.random.choice(total_scores, size=len(ground_truth_score), replace=True)
    
    # Replace the edge Score in the copied dataframe with the resampled Score
    inferred_network_df_copy["Score"] = resampled_inferred_network_scores
    ground_truth_df_copy["Score"] = resampled_ground_truth_scores

    # Calculate the lower threshold of the randomized scores(if not provided)
    if lower_threshold == None:
        randomized_ground_truth_mean = ground_truth_df_copy["Score"].mean()
        randomized_ground_truth_stdev = ground_truth_df_copy["Score"].std()
        
        # Default lower threshold is 1 stdev below the ground truth mean
        lower_threshold = randomized_ground_truth_mean - 1 * randomized_ground_truth_stdev
    
    # Recalculate the accuracy metrics for the resampled Score
    randomized_accuracy_metric_dict, randomized_confusion_matrix_dict = calculate_accuracy_metrics(
        ground_truth_df_copy,
        inferred_network_df_copy,
        lower_threshold,
        num_edges_for_early_precision
        )
    
    randomized_inferred_dict = {'Randomized': inferred_network_df_copy}
    randomized_ground_truth_dict = {'Randomized': ground_truth_df_copy}
    
    if histogram_save_path != None:
        print(f'\tSaving histogram of randomized GRN scores')
        plot_multiple_histogram_with_thresholds(randomized_ground_truth_dict, randomized_inferred_dict, histogram_save_path)
    
    return randomized_accuracy_metric_dict, randomized_confusion_matrix_dict

def calculate_and_plot_auroc_auprc(
    confusion_matrix_score_dict: dict,
    save_path: str
    ):
    """Plots the AUROC and AUPRC"""
    
    # Define figure and subplots for combined AUROC and AUPRC plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))    
    
    for method, score_dict in confusion_matrix_score_dict.items():
        print(f'\tGenerating AUROC and AUPRC for {method}')
        y_true = score_dict['y_true']
        y_scores = score_dict['y_scores']
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        prc_auc = auc(recall, precision)
        
        axes[0].plot(fpr, tpr, label=f'{method} AUROC = {roc_auc:.2f}')
        axes[0].plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
        
        axes[1].plot(recall, precision, label=f'{method} AUPRC = {prc_auc:.2f}')
        
        
    axes[0].set_title(f"{method.capitalize()} Combined AUROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
        
        
    axes[1].set_title(f"{method.capitalize()} Combined AUPRC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower right")

    # Adjust layout and display the figure
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def parse_wall_clock_time(line):
    # Extract the time part after the last mention of 'time'
    time_part = line.split("):")[-1].strip()
    
    # Split the time part by colons to get hours, minutes, and seconds if present
    time_parts = time_part.split(":")
    
    # Initialize hours, minutes, seconds to 0
    hours, minutes, seconds = 0, 0, 0
    
    # Clean up and parse each part
    if len(time_parts) == 3:  # h:mm:ss or h:mm:ss.ss
        hours = float(re.sub(r'[^\d.]', '', time_parts[0]))  # Remove non-numeric characters
        minutes = float(re.sub(r'[^\d.]', '', time_parts[1]))
        seconds = float(re.sub(r'[^\d.]', '', time_parts[2]))

    elif len(time_parts) == 2:  # m:ss or m:ss.ss
        minutes = float(re.sub(r'[^\d.]', '', time_parts[0]))
        seconds = float(re.sub(r'[^\d.]', '', time_parts[1]))

    # Calculate total time in seconds
    total_seconds = seconds + (minutes * 60) + (hours * 3600)
    hours = total_seconds * 0.0002778
    
    return hours

def parse_time_module_output(log_dir: str, sample_list: str):
    sample_resource_dict = {}
    
    samples = ["1", "2", "3", "4"]
    
    sample_list = [
        sample_dir for sample_dir in os.listdir(log_dir)
        if any(rep in sample_dir for rep in samples)
    ]

    for sample_log_dir in os.listdir(log_dir):
        print(f'Analyzing {sample_log_dir}')
        
        # Find each sample in the LOGS directory
        if sample_log_dir in sample_list:
            # Initialize pipeline_step_dict once per sample_log_dir
            sample_resource_dict[sample_log_dir] = {}
            
            # Find each step log file for the sample
            for file in os.listdir(f'{log_dir}/{sample_log_dir}'):
                
                if file.endswith(".log"):
                    print(file)
                    pipeline_step = file.split(".")[0]
                    sample_resource_dict[sample_log_dir][pipeline_step] = {
                        "user_time": 0,
                        "system_time": 0,
                        "percent_cpu": 0,
                        "wall_clock_time": 0,
                        "max_ram": 0
                    }

                    # Extract each relevant resource statistic for the sample step and save it in a dictionary
                    with open(f'{LOG_DIR}/{sample_log_dir}/{file}', 'r') as log_file:
                        for line in log_file:
                            if 'User time' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["user_time"] = float(line.split(":")[-1])
                            if 'System time' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["system_time"] = float(line.split(":")[-1])
                            if 'Percent of CPU' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["percent_cpu"] = float(line.split(":")[-1].split("%")[-2])
                            if 'wall clock' in line:
                                sample_resource_dict[sample_log_dir][pipeline_step]["wall_clock_time"] = parse_wall_clock_time(line)
                            if 'Maximum resident set size' in line:
                                kb_per_gb = 1048576
                                sample_resource_dict[sample_log_dir][pipeline_step]["max_ram"] = (float(line.split(":")[-1]) / kb_per_gb)

    return sample_resource_dict