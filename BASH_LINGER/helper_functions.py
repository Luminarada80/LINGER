import logging
import pandas as pd

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
        
        logging.info(f'\t{len(set(melted_df["Source"]))} TFs, {len(set(melted_df["Target"]))} TGs, and {len(melted_df["Score"])} edges')

    
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
    melted_df["Source"] = melted_df["Source"].str.upper().str.strip()
    melted_df["Target"] = melted_df["Target"].str.upper().str.strip()

    # Select and order columns as per new standard
    standardized_df = melted_df[["Source", "Target", "Score"]]
    
    logging.debug(f'\nNew df after standardizing:')
    logging.debug(standardized_df.head())
    
    return standardized_df

def find_ground_truth_scores_from_inferred(
    ground_truth: pd.DataFrame,
    inferred_network: pd.DataFrame,
    score_column_name: str = "Score"
    ) -> pd.DataFrame:
    
    """
    Merges the inferred network scores with the ground truth dataframe
    
    Parameters:
        ground_truth (pd.DataFrame):
            The ground truth dataframe. Columns should be "Source" and "Target"
            
        inferred_network (pd.DataFrame):
            The inferred GRN dataframe. Columns should be "Source", "Target", and "Score"
        
        score_column_name (str):
            Renames the "Score" column to a specific name, used if multiple datasets are being compared.
    
    """
    # Make sure that ground truth and inferred network "Source" and "Target" columns are uppercase
    ground_truth["Source"] = ground_truth["Source"].str.upper()
    ground_truth["Target"] = ground_truth["Target"].str.upper()
    
    inferred_network["Source"] = inferred_network["Source"].str.upper()
    inferred_network["Target"] = inferred_network["Target"].str.upper()
    
    # Take the Source and Target from the ground truth and the Score from the inferred network to create a new df
    ground_truth_with_scores = pd.merge(
        ground_truth, 
        inferred_network[["Source", "Target", "Score"]], 
        left_on=["Source", "Target"], 
        right_on=["Source", "Target"], 
        how="left"
    ).rename(columns={"Score": score_column_name})

    
    return ground_truth_with_scores