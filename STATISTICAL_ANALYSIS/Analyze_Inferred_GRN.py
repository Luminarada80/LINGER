import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import math
from copy import deepcopy
import os
import logging
import csv

# Install using 'conda install luminarada80::grn_analysis_tools' 
# or update to the newest version using 'conda update grn_analysis_tools'
from grn_analysis_tools import grn_formatting, plotting, resource_analysis, grn_stats

# Temporarily disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

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

def plot_resources_by_step(resource_dict: dict, output_dir: str):
    # Plot the resource requirements by step for each sample
    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='user_time',
        ylabel='User Time (h) / Percent CPU Usage',
        title='User Time / Percent CPU Usage by Step for Each Sample',
        filename=f'{output_dir}/Step_User_Time_Summary.png',
        divide_by_cpu=True
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='system_time',
        ylabel='System Time (h) / Percent CPU Usage',
        title='System Time / Percent CPU Usage by Step for Each Sample',
        filename=f'{output_dir}/Step_System_Time.png',
        divide_by_cpu=True
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='wall_clock_time',
        ylabel='Wall Clock Time (h)',
        title='Wall Clock Time by Step for Each Sample',
        filename=f'{output_dir}/Step_Wall_Clock_Time.png',
        divide_by_cpu=False
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='max_ram',
        ylabel='Max RAM Usage (GB)',
        title='Max RAM usage by Step for Each Sample',
        filename=f'{output_dir}/Step_Max_Ram.png',
        divide_by_cpu=False
    )

    plotting.plot_metric_by_step_adjusted(
        sample_resource_dict=resource_dict,
        metric='percent_cpu',
        ylabel='Percent CPU',
        title='Percent of the CPU Used',
        filename=f'{output_dir}/Step_Percent_Cpu.png',
        divide_by_cpu=False
    )

def plot_resources_by_sample(resource_dict: dict, output_dir: str):
    # Plot the resource requirements for running the entire pipeline
    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='user_time',
        ylabel='Total User Time / Percent CPU Usage',
        title='Total User Time / Percent CPU Usage for Each Sample',
        filename=f'{output_dir}/Total_User_Time.png',
        divide_by_cpu=True
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='system_time',
        ylabel='Total System Time / Percent CPU Usage',
        title='Total System Time / Percent CPU Usage for Each Sample',
        filename=f'{output_dir}/Total_System_Time.png',
        divide_by_cpu=True
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='wall_clock_time',
        ylabel='Wall Clock Time (h)',
        title='Total Wall Clock Time',
        filename=f'{output_dir}/Total_Wall_Clock_Time.png',
        divide_by_cpu=False
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='max_ram',
        ylabel='Max RAM Usage (GB)',
        title='Max RAM usage',
        filename=f'{output_dir}/Total_Max_Ram.png',
        divide_by_cpu=False
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='max_ram',
        ylabel='Max RAM Usage (GB)',
        title='Max RAM usage',
        filename=f'{output_dir}/Total_Max_Ram.png',
        divide_by_cpu=False
    )

    plotting.plot_total_metric_by_sample(
        sample_resource_dict=resource_dict,
        metric='percent_cpu',
        ylabel='Percent CPU',
        title='Average Percent of the CPU Used',
        filename=f'{output_dir}/Total_Percent_Cpu.png',
        divide_by_cpu=False
    )

def create_resource_requirement_summary(resource_dict: dict, output_dir: str) -> None:
    """
    Creates a summary file of each step for each sample from the output of the time module.

    Parameters
    ----------
        resource_dict (dict): 
            A dictionary containing the resources used for each step for each sample.
        output_dir (str): 
            The path to the directory in which to output the summary .tsv file
    """
    summary_dict = {}

    for sample, step_dict in resource_dict.items():
        if sample not in summary_dict:
            summary_dict[sample] = {
                    "user_time": 0,
                    "system_time": 0,
                    "percent_cpu": [],
                    "wall_clock_time": 0,
                    "max_ram": []
                }
        for step, sample_resource_dict in step_dict.items():
            for resource_name, resource_value in sample_resource_dict.items():
                if resource_name == "percent_cpu":
                    summary_dict[sample][resource_name].append(round(resource_value,2))
                elif resource_name == "max_ram":
                    summary_dict[sample][resource_name].append(round(resource_value,2))
                else:
                    summary_dict[sample][resource_name] += round(resource_value,2)
        summary_dict[sample]["max_ram"] = max(summary_dict[sample]["max_ram"])
        summary_dict[sample]["percent_cpu"] = round(sum(summary_dict[sample]["percent_cpu"]) / len(summary_dict[sample]["percent_cpu"]),2)
        
    summary_df = pd.DataFrame(summary_dict)
    summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)
    print(summary_df.head())

    summary_df.to_csv(f'{output_dir}/Resource_Summary.tsv', sep='\t')

def read_input_files() -> tuple[str, list, list, dict]:
    """
    Reads through the current directory to find input files.
    
    Requires a folder called "INPUT", which contains a subfolder for each 
    method. Each method folder should have a subfolder for each sample, containing
    the inferred network for that sample.
    
    The ground truth file should be stored in a separate file within "INPUT" called
    "GROUND_TRUTH".

    Returns
    ----------
        ground_truth_path (str):
            The full path to the ground truth file.
        method_names (list):
            A list containing the method names (names of the subfolders of "INPUT")
        sample_names (list):
            A list containing the sample names (names of the subfolders of each method directory)
        inferred_network_dict (dict):
            A dictionary of the paths to the inferred network files, with the method name and sample
            name corresonding to the file as the keys to the dictionary.
    """
    # Read through the directory to find the input and output files
    method_names = []
    inferred_network_dict = {}
    sample_names = set()
    print(f'\n---- Directory Structure ----')
    
    # Looks through the current directory for a folder called "input"
    for folder in os.listdir("."):
        if folder.lower() == "input":
            print(folder)
            for subfolder in os.listdir(f'./{folder}'):
                print(f'  └──{subfolder}')
                
                # Finds the folder titled "ground_truth"
                if subfolder.lower() == "ground_truth":
                    ground_truth_dir = f'{os.path.abspath(folder)}/{subfolder}' 
                    
                    # Check to see if there are more than 1 ground truth files
                    num_ground_truth_files = len(os.listdir(ground_truth_dir))
                    if num_ground_truth_files > 1:
                        logging.warning(f'Warning! Multiple files in ground truth input directory. Using the first entry...')
                        
                    # Use the first file in the ground truth directory as the ground truth and set the path
                    ground_truth_filename = os.listdir(ground_truth_dir)[0]
                    ground_truth_path = f"{ground_truth_dir}/{ground_truth_filename}"
                    for file_name in os.listdir(ground_truth_dir):
                        print(f'    └──{file_name}')
                    
                # If the folder is not "ground_truth", assume the folders are the method names
                else:
                    # Create output directories for the samples
                    if not os.path.exists(f'./OUTPUT/{subfolder}'):
                        os.makedirs(f'./OUTPUT/{subfolder}')
                    
                    # Set the method name to the name of the subfolder
                    method_name = subfolder
                    method_names.append(method_name)
                    method_input_dir = f'{os.path.abspath(folder)}/{subfolder}' 
                    
                    # Add the method as a key for the inferred network dictionary
                    if method_name not in inferred_network_dict:
                        inferred_network_dict[method_name] = {}

                    # Iterate through each sample folder in the method directory
                    for sample_name in os.listdir(method_input_dir):
                        
                        sample_dir_path = f'{method_input_dir}/{sample_name}'
                        
                        # If there is a file in the sample folder
                        if len(os.listdir(sample_dir_path)) > 0:
                            print(f'     └──{sample_name}')
                            
                            # Add the name of the folder as a sample name
                            sample_names.add(sample_name)
                            
                            # Create output directories for the samples
                            if not os.path.exists(f'./OUTPUT/{subfolder}/{sample_name}'):
                                os.makedirs(f'./OUTPUT/{subfolder}/{sample_name}')
                            
                            # Add the path to the sample to the inferred network dictionary for the method
                            if sample_name not in inferred_network_dict[method_name]:
                                inferred_network_dict[method_name][sample_name] = None
                            
                            # Find the path to the inferred network file for the current sample
                            num_sample_files = len(os.listdir(sample_dir_path))
                            if num_sample_files > 1:
                                logging.warning(f"Warning! Multiple files in the inferred network directory for {sample_name}. Using the first entry...")
                            for inferred_network_file in os.listdir(sample_dir_path):
                                inferred_network_dict[method_name][sample_name] = f'{method_input_dir}/{sample_name}/{inferred_network_file}'
                                print(f'        └──{inferred_network_file}')
    
    print(f'\nSample names:')
    for sample_name in sample_names:
        print(f'\t{sample_name}')
        
    print(f'\nMethod names:')
    for method_name in method_names:
        print(f'\t{method_name}')
    
    print(f'\nGround truth file:')
    print(f'\t{ground_truth_filename}')
    
    return ground_truth_path, method_names, sample_names, inferred_network_dict

def load_inferred_network_df(
    inferred_network_file: str,
    separator: str
    ) -> pd.DataFrame:
    """
    Loads the inferred network file to a dataframe based on the separator.

    Parameters
    ----------
        inferred_network_file (str):
            Path to the inferred network file.
        separator (str):
            Pandas separator to use when loading in the dataframe
    """
    return pd.read_csv(inferred_network_file, sep=separator, index_col=0, header=0)

def standardize_ground_truth_format(ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes Source and Target gene names to uppercase and removes any whitespace

    Parameters
    ----------
        ground_truth_df (pd.DataFrame): 
            Unfiltered ground truth dataframe

    Returns
    ----------
        ground_truth_df (pd.DataFrame):
            Ground truth dataframe with Source and Target names converted to uppercase
    """
    ground_truth_df['Source'] = ground_truth_df['Source'].str.upper().str.strip()
    ground_truth_df['Target'] = ground_truth_df['Target'].str.upper().str.strip()
    
    return ground_truth_df

def create_ground_truth_copies(
    ground_truth_df: pd.DataFrame,
    method_names: list,
    sample_names: list,
    inferred_network_dict: dict
    ) -> dict:
    """
    Creates copies of the ground truth dataframe for each method and sample.
    
    Each inferred network needs a corresponding ground truth dataframe, as the ground truth edge
    scores are assigned from the inferred network

    Parameters
    ----------
        ground_truth_df (pd.DataFrame): 
            Ground truth dataframe with columns "Source" and "Target"
        method_names (list): 
            A list of the names of each inference method being evaluated.
        sample_names (list): 
            A list of the dataset sample names.
        inferred_network_dict (dict): 
            A dictionary of each inferred network for the method, used to find sample names
            for the current method

    Returns
    ----------
        ground_truth_dict (dict):
            A dictionary of ground truth dataframe copies, mapped to a sample and method
    """
    ground_truth_dict = {}
    for method in method_names:
        ground_truth_dict[method] = {}
        
        method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
        for sample in method_samples:
            ground_truth_dict[method][sample] = deepcopy(ground_truth_df)
    
    return ground_truth_dict

def write_method_accuracy_metric_file(total_accuracy_metric_dict: dict) -> None:
    """
    For each inference method, creates a pandas dataframe of the accuracy metrics for each sample and 
    outputs the results to a tsv file.
    
    total_accuracy_metric_dict[metric][sample][accuracy_metric_name] = score

    Parameters
    ----------
        total_accuracy_metric_dict (dict): 
            A dictionary containing the method names, sample names, accuracy metric names, and accuracy 
            metric values.
    """
    for method in total_accuracy_metric_dict.keys():
        total_accuracy_metrics_df = pd.DataFrame(total_accuracy_metric_dict[method]).T
        total_accuracy_metrics_df.to_csv(f'OUTPUT/{method.lower()}_total_accuracy_metrics.tsv', sep='\t')

def update_metrics(
    confusion_score_dict: dict,
    randomized_dict: dict,
    method: str,
    confusion_matrix_dict: dict,
    randomized_confusion_matrix_dict: dict
    ) -> None:
    """
    Adds y_true and y_scores to the total method dictionary used for generating the AUROC and AUPRC curves.

    Parameters
    ----------
        confusion_score_dict (dict):
            A dictionary containing lists of y_true and y_scores for each sample in a method.
            Used to generate multiple AUROC and AUPRC curves on the same plot for multiple methods.
        randomized_dict (dict):
            A dictionary of randomized y_true and y_scores values between the ground truth and inferred
            network, to show how much better the inference method performs above random. 
        method (str):
            The name of the inference method. 
        confusion_matrix_dict (dict):
            A dictionary containing the confusion matrix, y_true array, and y_scores array. Generated by 
            grn_stats.calculate_accuracy_metrics.
        randomized_confusion_matrix_dict (dict): 
            A dictionary containing the confusion matrix, y_true array, and y_scores array for the randomized
            values. Generated by grn_stats.create_randomized_inference_scores.
    """
    # Update scores for plotting
    confusion_score_dict[method]["y_true"].append(confusion_matrix_dict["y_true"])
    confusion_score_dict[method]["y_scores"].append(confusion_matrix_dict["y_scores"])

    randomized_dict[f"{method} Original"]["y_true"].append(confusion_matrix_dict["y_true"])
    randomized_dict[f"{method} Original"]["y_scores"].append(confusion_matrix_dict["y_scores"])
    randomized_dict[f"{method} Randomized"]["y_true"].append(randomized_confusion_matrix_dict["y_true"])
    randomized_dict[f"{method} Randomized"]["y_scores"].append(randomized_confusion_matrix_dict["y_scores"])

def log_auroc_auprc(
    confusion_matrix_dict: dict,
    randomized_confusion_dict: dict
    ) -> tuple[float, float, float, float]:
    """
    Calculates the normal and randomized AUROC and AUPRC values from
    confusion_matrix_dict and randomized_confusion_dict.

    Parameters
    ----------
        confusion_matrix_dict (_type_): _description_
        randomized_confusion (_type_): _description_

    Return
    ----------
        auroc (float):
            The AUROC value for the sample.
        randomized_auroc (float): 
            The AUROC value for the randomized edge scores for the sample.
        auprc (float):
            The AUPRC value for the sample.
        randomized_auprc (float):
            The AUPRC value for the randomized edge scores for the sample
    """
    auroc = grn_stats.calculate_auroc(confusion_matrix_dict)
    randomized_auroc = grn_stats.calculate_auroc(randomized_confusion_dict)

    auprc = grn_stats.calculate_auprc(confusion_matrix_dict)
    randomized_auprc = grn_stats.calculate_auprc(randomized_confusion_dict)

    return auroc, randomized_auroc, auprc, randomized_auprc

def plot_sample_metrics(
    method: str,
    sample: str,
    confusion_matrix_dict: dict,
    randomized_confusion_dict: dict
    ) -> None:
    """
    Plots both the normal AUROC and AUPRC for the sample, and the normal vs randomized AUROC and AUPRC.

    Args:
        method (str): 
            The name of the inference method.
        sample (str): 
            The name of the sample
        confusion_matrix_dict (dict): 
            The confusion matrix dictionary for the sample with y_true and y_scores (from 
            grn_stats.calculate_accuracy_metrics)
        randomized_confusion_dict (dict): 
            The confusion matrix dictionary for the sample with y_true and y_scores (from 
            grn_stats.create_randomized_inference_scores)
    """
    # Plot normal AUROC and AUPRC
    confusion_with_method = {method: confusion_matrix_dict}
    plotting.plot_auroc_auprc(confusion_with_method, f"./OUTPUT/{method}/{sample}/auroc_auprc.png")

    # Plot randomized AUROC and AUPRC
    randomized_with_method = {
        f"{method} Original": {"y_true": confusion_matrix_dict["y_true"], "y_scores": confusion_matrix_dict["y_scores"]},
        f"{method} Randomized": {"y_true": randomized_confusion_dict["y_true"], "y_scores": randomized_confusion_dict["y_scores"]},
    }
    plotting.plot_auroc_auprc(randomized_with_method, f"./OUTPUT/{method}/{sample}/randomized_auroc_auprc.png")

def write_metrics_to_file(
    method: str,
    sample: str,
    accuracy_metrics: dict,
    randomized_metrics: dict
    ) -> None:
    """
    Writes the accuracy metrics for the normal and randomized edge scores to tsv files in the sample output directory.

    Parameters
    ----------
        method (str):
            The name of the inference method.
        sample (str):
            The name of the sample.
        accuracy_metrics (dict): 
            A dictionary of accuracy metric names with resulting scores.
        randomized_metrics (dict): 
            A dictionary of the accuracy metrics calculated from the randomized edge scores.
    """
    # Write normal accuracy metrics
    with open(f"./OUTPUT/{method.upper()}/{sample.upper()}/accuracy_metrics.tsv", "w") as file:
        file.write("Metric\tScore\n")
        for metric_name, score in accuracy_metrics.items():
            file.write(f"{metric_name}\t{score:.4f}\n")

    # Write randomized accuracy metrics
    with open(f"./OUTPUT/{method.upper()}/{sample.upper()}/randomized_accuracy_metrics.tsv", "w") as file:
        file.write("Metric\tOriginal Score\tRandomized Score\n")
        for metric_name, score in accuracy_metrics.items():
            file.write(f"{metric_name}\t{score:.4f}\t{randomized_metrics[metric_name]:.4f}\n")

def update_accuracy_metrics(
    total_metrics, random_metrics,
    method, sample,
    auroc, auprc, randomized_auroc, randomized_auprc,
    confusion_matrix, randomized_confusion_matrix
):
    """
    Updates the total and randomized accuracy metric dictionaries for a given method and sample.

    Parameters:
    ----------
    total_metrics : dict
        Dictionary to store the original metrics for each method and sample.
    random_metrics : dict
        Dictionary to store the randomized metrics for each method and sample.
    method : str
        The name of the method being processed.
    sample : str
        The name of the sample being processed.
    auroc : float
        AUROC for the original data.
    auprc : float
        AUPRC for the original data.
    randomized_auroc : float
        AUROC for the randomized data.
    randomized_auprc : float
        AUPRC for the randomized data.
    confusion_matrix : dict
        Confusion matrix for the original data.
    randomized_confusion_matrix : dict
        Confusion matrix for the randomized data.
    """
    # Ensure nested dictionaries exist for the method and sample
    if method not in total_metrics:
        total_metrics[method] = {}
    if sample not in total_metrics[method]:
        total_metrics[method][sample] = {}

    if method not in random_metrics:
        random_metrics[method] = {}
    if sample not in random_metrics[method]:
        random_metrics[method][sample] = {}

    # Update total accuracy metrics
    total_metrics[method][sample].update({
        "auroc": auroc,
        "auprc": auprc,
        "true_positive": int(confusion_matrix["true_positive"]),
        "true_negative": int(confusion_matrix["true_negative"]),
        "false_positive": int(confusion_matrix["false_positive"]),
        "false_negative": int(confusion_matrix["false_negative"]),
    })

    # Update randomized accuracy metrics
    random_metrics[method][sample].update({
        "auroc": randomized_auroc,
        "auprc": randomized_auprc,
        "true_positive": int(randomized_confusion_matrix["true_positive"]),
        "true_negative": int(randomized_confusion_matrix["true_negative"]),
        "false_positive": int(randomized_confusion_matrix["false_positive"]),
        "false_negative": int(randomized_confusion_matrix["false_negative"]),
    })

def preprocess_inferred_and_ground_truth_networks(ground_truth_path, method_names, sample_names, inferred_network_dict):
    """
    Preprocesses the ground truth and inferred networks for each sample in each method, 
    applying each processing step to all samples before moving to the next step.
    
    Uses grn_formatting from the luminarada80::grn_analysis_tools package to create standard dataframes for each inferred
    network. The standard format is a pd.DataFrame object with columns "Source" (TFs), "Target", (TGs), and "Score" in long
    format.
    
    Once the ground truth and inferred networks are in the correct format, edge scores are added to the ground truth from the 
    inferred network by locating the same "Source" "Target" pairs in the inferred network that are present in the ground truth
    network.
    
    Any rows containing TFs or TGs not present in the ground truth network are removed from the inferred network, so that the 
    only TF-TG scores being compared are present in both networks. 
    
    TF-TG pairs present in the ground truth are removed from the inferred network, so that the inferred network does not have
    any overlapping edges with the ground truth. 
    
    The scores are then evaluated by setting a lower threshold based on the ground truth edge scores. Scores above the threshold
    in the ground truth dataset are used as true positives, while scores below this threshold in the ground truth dataset are 
    set to false negatives (this assumes that the ground truth dataset represents the full list of true interactions between the
    TFs and TGs present in the ground truth). In the inferred network, scores above the threshold are used as false positives (the
    edges are scored highly but not in the ground truth) while scores below the threshold are used as true negatives (the edges 
    are correctly scored lower than the threshold and are not in the ground truth).
    
    Parameters
    ----------
    ground_truth_path (str):
        The path to the ground truth file.
    method_names (list):
        A list of the inference method names being compared.
    sample_names (list):
        A list of the sample names being evaluated.
    inferred_network_dict (dict):
        A dictionary of the paths to the inferred network files. 
        `inferred_network_dict[method][sample] = file_path`
    
    Returns
    ----------
        processed_ground_truth_dict (dict):
            A dictionary of processed ground truth dataframes. 
            `processed_ground_truth_dict[method][sample] = processed_ground_truth_df`
        processed_inferred_network_dict (dict):
            A dictionary of processed inferred network dataframes. 
            `processed_inferred_network_dict[method][sample] = processed_inferred_network_df`
    """
    print(f'\tReading ground truth')
    ground_truth = pd.read_csv(ground_truth_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = standardize_ground_truth_format(ground_truth)
    ground_truth_dict = create_ground_truth_copies(ground_truth, method_names, sample_names, inferred_network_dict)
    
    processed_inferred_network_dict: dict = {}
    processed_ground_truth_dict: dict = {}

    # Initial setup: Load and standardize inferred network dataframes
    inferred_networks = {}
    for method in method_names:
        print(f'\tLoading and standardizing inferred networks for {method}')
        method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
        inferred_networks[method] = {}
        for sample in method_samples:
            inferred_network_file = inferred_network_dict[method][sample]
            sep = ',' if method == 'CELL_ORACLE' else '\t'
            inferred_network_df = load_inferred_network_df(inferred_network_file, sep)
            
            if method == "CELL_ORACLE":
                inferred_network_df = grn_formatting.create_standard_dataframe(
                    inferred_network_df, source_col='source', target_col='target', score_col='coef_abs'
                )
            else:
                inferred_network_df = grn_formatting.create_standard_dataframe(
                    inferred_network_df, source_col='Source', target_col='Target', score_col='Score'
                )
            inferred_networks[method][sample] = inferred_network_df

    # Processing steps
    print(f'\nPreprocessing:')
    for step, description in enumerate([
        "Adding inferred scores to ground truth",
        "Setting scores to log2",
        "Removing genes from inferred not in ground truth",
        "Removing ground truth edges from inferred",
        "Removing NaN values from the inferred network",
        "Classifying interactions by ground truth threshold"
    ]):
        print(f"\tStep {step+1}: {description}")
        for method in method_names:
            method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
            for sample in method_samples:
                sample_ground_truth = ground_truth_dict[method][sample]
                inferred_network_df = inferred_networks[method][sample]
                
                if description == "Adding inferred scores to ground truth":
                    sample_ground_truth = grn_formatting.add_inferred_scores_to_ground_truth(
                        sample_ground_truth, inferred_network_df
                    )
                    
                elif description == "Setting scores to log2":
                    inferred_network_df["Score"] = np.log2(inferred_network_df["Score"])
                    sample_ground_truth["Score"] = np.log2(sample_ground_truth["Score"])
                    
                elif description == "Removing genes from inferred not in ground truth":
                    inferred_network_df = grn_formatting.remove_tf_tg_not_in_ground_truth(
                        sample_ground_truth, inferred_network_df
                    )
                    
                elif description == "Removing ground truth edges from inferred":
                    inferred_network_df = grn_formatting.remove_ground_truth_edges_from_inferred(
                        sample_ground_truth, inferred_network_df
                    )
                    
                elif description == "Removing NaN values from the inferred network":
                    inferred_network_df = inferred_network_df.dropna(subset=['Score'])
                    sample_ground_truth = sample_ground_truth.dropna(subset=['Score'])
                    
                elif description == "Classifying interactions by ground truth threshold":
                    sample_ground_truth, inferred_network_df = grn_stats.classify_interactions_by_threshold(
                        sample_ground_truth, inferred_network_df
                    )
                
                # Update processed data
                ground_truth_dict[method][sample] = sample_ground_truth
                inferred_networks[method][sample] = inferred_network_df

    # Finalize processed data
    for method in method_names:
        processed_inferred_network_dict[method] = {}
        processed_ground_truth_dict[method] = {}
        method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
        for sample in method_samples:
            processed_inferred_network_dict[method][sample] = inferred_networks[method][sample]
            processed_ground_truth_dict[method][sample] = ground_truth_dict[method][sample]

    return processed_ground_truth_dict, processed_inferred_network_dict

def main():
    print(f"Reading input files")
    ground_truth_path, method_names, sample_names, inferred_network_dict = read_input_files()

    print(f"\nLoading and Standardizing networks")
    processed_ground_truth_dict, processed_inferred_network_dict = preprocess_inferred_and_ground_truth_networks(
        ground_truth_path, method_names, sample_names, inferred_network_dict
    )

    total_method_confusion_scores = {}
    total_accuracy_metrics = {}
    random_accuracy_metrics = {}

    for method, sample_dict in processed_inferred_network_dict.items():
        print(f"\nCalculating Statistics for {method}")
        total_method_confusion_scores[method] = {"y_true": [], "y_scores": []}
        randomized_method_dict = {
            f"{method} Original": {"y_true": [], "y_scores": []},
            f"{method} Randomized": {"y_true": [], "y_scores": []},
        }
        total_accuracy_metrics[method] = {}
        random_accuracy_metrics[method] = {}

        # Logging steps for processing
        for step, description in enumerate([
            "Computing accuracy metrics and confusion metrics",
            "Computing randomized metrics",
            "Saving metrics to output files",
            "Computing AUROC and AUPRC",
            "Updating confusion scores and accuracy metrics",
            "Plotting AUROC and AUPRC metrics for the sample",
        ]):
            print(f"\tStep {step+1}: {description}")
            for sample in sample_dict:
                # Perform the specific step
                if description == "Computing accuracy metrics and confusion metrics":
                    accuracy_metrics, confusion_matrix = grn_stats.calculate_accuracy_metrics(
                        processed_ground_truth_dict[method][sample],
                        processed_inferred_network_dict[method][sample]
                        )
                    
                elif description == "Computing randomized metrics":
                    randomized_metrics, randomized_confusion = grn_stats.create_randomized_inference_scores(
                        processed_ground_truth_dict[method][sample],
                        processed_inferred_network_dict[method][sample]
                    )
                    
                elif description == "Saving metrics to output files":
                    write_metrics_to_file(
                        method, sample, accuracy_metrics, randomized_metrics
                    )
                    
                elif description == "Computing AUROC and AUPRC":
                    auroc, randomized_auroc, auprc, randomized_auprc = log_auroc_auprc(
                        confusion_matrix, randomized_confusion
                    )
                    
                elif description == "Updating confusion scores and accuracy metrics":
                    update_metrics(
                        total_method_confusion_scores, randomized_method_dict,
                        method, confusion_matrix, randomized_confusion
                    )
                    
                    update_accuracy_metrics(
                        total_accuracy_metrics, random_accuracy_metrics,
                        method, sample, auroc, auprc, randomized_auroc, randomized_auprc,
                        confusion_matrix, randomized_confusion
                    )
                    
                elif description == "Plotting AUROC and AUPRC metrics for the sample":
                    plot_sample_metrics(
                        method, sample, confusion_matrix, randomized_confusion
                    )

        # Plot aggregated metrics for the current method
        print(f"\tStep 7: Plotting {method.lower()} original vs randomized AUROC and AUPRC for all samples")
        plotting.plot_multiple_method_auroc_auprc(
            randomized_method_dict, f"./OUTPUT/{method.lower()}_randomized_auroc_auprc.png"
        )

    # Plot combined metrics across all methods
    print(f"\nPlotting AUROC and AUPRC comparing all methods")
    plotting.plot_multiple_method_auroc_auprc(
        total_method_confusion_scores, "./OUTPUT/auroc_auprc_combined.png"
    )

    # Save final accuracy metrics
    print(f"Saving final accuracy metrics")
    write_method_accuracy_metric_file(total_accuracy_metrics)
    write_method_accuracy_metric_file(random_accuracy_metrics)
    
    print(f'Done! Successfully completed analysis')

                
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')  
    
    main()