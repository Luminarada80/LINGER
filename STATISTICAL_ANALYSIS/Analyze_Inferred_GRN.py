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

def create_resource_requirement_summary(resource_dict: dict, output_dir: str):
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

def read_input_files():
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
        
    print(f'\nInferred network files:')
    for method, sample_dict in inferred_network_dict.items():
        print(f'\tInferred network {method}')
        for sample_name, inferred_network_path in sample_dict.items():
            print(f'\t\t{sample_name} = {inferred_network_path}')
    
    print(f'\nGround truth file:')
    print(f'\t{ground_truth_path}')
    
    return ground_truth_path, method_names, sample_names, inferred_network_dict

def standardize_inferred_network_dataframe_format(method, sample, inferred_network_dict):
    # Retrieve the path to the inferred network for the current sample and method
    logging.debug(inferred_network_dict[method])
    inferred_network_file = inferred_network_dict[method][sample]

    # Load the inferred dataset for the sample
    if method == "CELL_ORACLE":
        inferred_network_df = pd.read_csv(inferred_network_file, sep=',', index_col=0, header=0)
        inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df, source_col='source', target_col='target', score_col='coef_abs')
    else:
        inferred_network_df = pd.read_csv(inferred_network_file, sep='\t', index_col=0, header=0)
        logging.debug(f'\nInferred_network_df BEFORE standardization:')
        logging.debug(inferred_network_df.head())
        inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df)
        logging.debug(f'Inferred Net: {len(set(inferred_network_df["Source"])):,} TFs, {len(set(inferred_network_df["Target"])):,} TGs, and {len(inferred_network_df["Source"]):,} edges\n')

        logging.debug(f'\nInferred_network_df AFTER standardization:')
        logging.debug(inferred_network_df.head())
    
    return inferred_network_df

def add_accuracy_metrics_to_method_dict(accuracy_metric_dict, method):
    accuracy_metric_dict[method] = {
            'sample_name': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'accuracy': [],
            'f1_score': [],
            'jaccard_index': [],
            'weighted_jaccard_index': [],
            'early_precision_rate': [],
            'auroc': [],
            'auprc': []
        }
    return accuracy_metric_dict

def standardize_ground_truth_format(ground_truth_df):
    ground_truth_df['Source'] = ground_truth_df['Source'].str.upper().str.strip()
    ground_truth_df['Target'] = ground_truth_df['Target'].str.upper().str.strip()
    
    return ground_truth_df

def create_ground_truth_copies(ground_truth, method_names, sample_names, inferred_network_dict):
    ground_truth_dict = {}
    for method in method_names:
        ground_truth_dict[method] = {}
        
        method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
        for sample in method_samples:
            ground_truth_dict[method][sample] = deepcopy(ground_truth)
    
    return ground_truth_dict

def main():
    ground_truth_path, method_names, sample_names, inferred_network_dict = read_input_files()
    
    # Read in and standardize the ground truth dataframe
    ground_truth = pd.read_csv(ground_truth_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = standardize_ground_truth_format(ground_truth)
    ground_truth_dict = create_ground_truth_copies(ground_truth, method_names, sample_names, inferred_network_dict)
    
    total_accuracy_metrics: dict = {}
    random_accuracy_metrics: dict = {}   
    method_inferred_net_dict: dict = {}
    confusion_matrix_dict_all_methods: dict = {}

    # Iterate through each sample in the list of sample names / numbers
    logging.info(f'\n----- Inferred Networks vs Ground Truth Statistical Analysis -----')
    for method in method_names:        
        # Find the samples that are in the method
        method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
            
        if method not in method_inferred_net_dict:
            method_inferred_net_dict[method] = {}
        
        # Iterate through each sample for the current method
        for i, sample in enumerate(method_samples):
            logging.info(f'\n{method} {sample} ({i+1}/{len(method_samples)})')
            
            logging.info(f'\tPreprocessing:')
            # Add the inferred dataset for the sample to the method
            logging.info(f'\t\t1/4: Standardizing the inferred network dataframe format')
            inferred_network_df = standardize_inferred_network_dataframe_format(method, sample, inferred_network_dict)
            
            method_inferred_net_dict[method][sample] = inferred_network_df
                
            # Create a dictionary to store the individual ground truth scores for the method and sample                
            
            
            # Process the ground truth and inferred networks
            logging.debug('\n----- Processing inferred networks and ground truths -----')                
            sample_result_dir = f'OUTPUT/{method}/{sample}'
                
            if not os.path.exists(sample_result_dir):
                os.makedirs(sample_result_dir)
            
            # Write out the network size for the ground truth
            with open(f'{sample_result_dir}/summary_statistics.txt', 'w') as summary_stat_file:
                summary_stat_file.write(f'Dataset\tTFs\tTGs\tEdges\n')
                summary_stat_file.write(f'Ground_truth\t{len(set(ground_truth["Source"]))}\t{len(set(ground_truth["Target"]))}\t{len(ground_truth["Source"])}\n')
                
            logging.debug(f'Ground truth: {len(set(ground_truth["Source"])):,} TFs, {len(set(ground_truth["Target"])):,} TGs, and {len(ground_truth["Source"]):,} edges\n')
            
            # Add the method and samples to the total and random accuracy metrics dictionary
            if method not in total_accuracy_metrics:
                total_accuracy_metrics = add_accuracy_metrics_to_method_dict(total_accuracy_metrics, method)
            
            if method not in random_accuracy_metrics:
                random_accuracy_metrics = add_accuracy_metrics_to_method_dict(random_accuracy_metrics, method)
            
            total_accuracy_metrics[method]['sample_name'].append(sample)
            random_accuracy_metrics[method]['sample_name'].append(sample)
            
            # Append the inferred network information to the summary stats file
            with open(f'{sample_result_dir}/summary_statistics.txt', 'a') as summary_stat_file:
                logging.debug(inferred_network_df.columns)
                summary_stat_file.write(f'{method.capitalize()}\t{len(set(inferred_network_df["Source"]))}\t{len(set(inferred_network_df["Target"]))}\t{len(inferred_network_df["Source"])}\n')
            
            # Create a copy of the ground truth dataframe for the method                
            method_ground_truth = ground_truth_dict[method][sample]
            
            logging.info(f'\t\t2/4: Retrieving scores for ground truth edges from the inferred network')
            method_ground_truth = grn_formatting.add_inferred_scores_to_ground_truth(method_ground_truth, inferred_network_df)
            
            logging.debug(f'\nGround truth for {method} BEFORE dropping NaN values')
            logging.debug(method_ground_truth.head())
            logging.debug(f'Size: {method_ground_truth.size}')
            
            # Drop any NaN scores in the ground truth after adding scores
            method_ground_truth = method_ground_truth.dropna(subset=['Score'])
            
            logging.debug(f'\nGround truth for {method} AFTER dropping NaN values')
            logging.debug(method_ground_truth.head())
            logging.debug(f'Size: {method_ground_truth.size}')
            
            # Set the inferred network and ground truth scores to log2 scale
            inferred_network_df["Score"] = np.log2(inferred_network_df["Score"])
            method_ground_truth["Score"] = np.log2(method_ground_truth["Score"])
            
            # Remove any TFs and TGs from the inferred network that are not in the ground truth network
            logging.info(f'\t\t3/4: Removing TFs and TGs from the inferred network that are not in ground truth')
            inferred_network_only_shared_tf_tg_df = grn_formatting.remove_tf_tg_not_in_ground_truth(method_ground_truth, inferred_network_df)
            logging.debug(f"\t\tGround truth shape: TFs = {len(set(method_ground_truth['Source']))}, TGs = {len(set(method_ground_truth['Target']))}, edges = {len(set(method_ground_truth['Score']))}")
            logging.debug(f"\t\tInferred same genes: TFs = {len(set(inferred_network_only_shared_tf_tg_df['Source']))}, TGs = {len(set(inferred_network_only_shared_tf_tg_df['Target']))}, edges = {len(set(inferred_network_only_shared_tf_tg_df['Score']))}")
            
            # Remove ground truth edges from the inferred network
            logging.info(f'\t\t4/4: Removing any ground truth edges from the inferred network')
            inferred_network_no_ground_truth_df = grn_formatting.remove_ground_truth_edges_from_inferred(method_ground_truth, inferred_network_only_shared_tf_tg_df)
            logging.debug(f"\t\tInferred no ground truth: TFs = {len(set(inferred_network_no_ground_truth_df['Source']))}, TGs = {len(set(inferred_network_no_ground_truth_df['Target']))}, edges = {len(set(inferred_network_no_ground_truth_df['Score']))}")
            
            # Drop any NaN scores in the inferred network after adding scores
            inferred_network_no_ground_truth_df = inferred_network_no_ground_truth_df.dropna(subset=['Score'])
            
            logging.info(f'\n\tStatistical Analysis')
            # Classify the true and predicted interactions for the ground truth and inferred networks
            logging.info(f'\t\t1/5: Classifying true and predicted interactions for the ground truth and inferred networks')
            method_ground_truth, inferred_network_no_ground_truth_df = grn_stats.classify_interactions_by_threshold(method_ground_truth, inferred_network_no_ground_truth_df)
            
            logging.info(f'\t\t2/5: Calculating accuracy metrics')
            # Calculate the accuracy metrics and the confusion matrix (TP, FP, TN, FN)
            accuracy_metric_dict, confusion_matrix_score_dict = grn_stats.calculate_accuracy_metrics(method_ground_truth, inferred_network_no_ground_truth_df)
            confusion_matrix_dict_all_methods[method][sample] = confusion_matrix_score_dict        
            
            # Calculate the accuracy metrics and confusion matrix when the edge scores for the ground truth and inferred networks are randomized
            logging.info(f'\t\t3/5: Calculating randomized accuracy metrics')
            randomized_histogram_path = f'{sample_result_dir}/Histogram_Randomized_GRN_Scores'
            randomized_accuracy_metric_dict, randomized_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(
                method_ground_truth,
                inferred_network_no_ground_truth_df,
                histogram_save_path=randomized_histogram_path
                )
            
            # Create a dictionary of the accuracy metrics for the randomized accuracy metrics and the original accuracy metrics
            randomized_method_dict = {
                f"{method} Original": confusion_matrix_score_dict,
                f"{method} Randomized": randomized_confusion_matrix_dict
            }
            
            # Calculate the AUROC and AUPRC for the randomized edge scores
            randomized_auroc = grn_stats.calculate_auroc(randomized_confusion_matrix_dict)
            randomized_auprc = grn_stats.calculate_auprc(randomized_confusion_matrix_dict)
            
            random_accuracy_metrics[method]["auroc"].append(randomized_auroc)
            random_accuracy_metrics[method]["auprc"].append(randomized_auprc)
            
            # Calculate the AUROC and AUPRC for the randomized and original edge scores
            logging.debug(f'\t\tGenerating AUROC and AUPRC comparing the randomized scores to the original scores')
            randomized_auc_path = f'{sample_result_dir}/{method}_randomized_auroc_auprc.png'
            plotting.plot_auroc_auprc(randomized_method_dict, randomized_auc_path)
            
            logging.debug(f'\n----- Accuracy Metrics -----')
            logging.debug(f'\tSaving accuracy metrics for {method}')
            with open(f'{sample_result_dir}/accuracy_metrics.tsv', 'w') as accuracy_metric_file:
                accuracy_metric_file.write(f'Metric\tScore\n')
                for metric_name, score in accuracy_metric_dict.items():
                    accuracy_metric_file.write(f'{metric_name}\t{score:.4f}\n')
                    total_accuracy_metrics[method][metric_name].append(score)
            
            logging.debug(f'\tSaving randomized score accuracy methods for {method}')
            with open(f'{sample_result_dir}/randomized_accuracy_method.tsv', 'w') as random_accuracy_file:
                random_accuracy_file.write(f'Metric\tOriginal Score\tRandomized Score\n')
                for metric_name, score in accuracy_metric_dict.items():
                    random_accuracy_file.write(f'{metric_name}\t{score:.4f}\t{randomized_accuracy_metric_dict[metric_name]:4f}\n')
                    random_accuracy_metrics[method][metric_name].append(randomized_accuracy_metric_dict[metric_name])
            
            ground_truth_dict[method][sample] = method_ground_truth
            method_inferred_net_dict[method][sample] = inferred_network_no_ground_truth_df
        
        for method in method_names:
            logging.info(f'\t\t4/5: Creating histogram of ground truth vs inferred GRN scores with threshold')
            histogram_path = f"{sample_result_dir}/Histogram_GRN_Scores_With_Threshold"
            plotting.plot_multiple_histogram_with_thresholds(ground_truth_dict[method], method_inferred_net_dict[method], histogram_path)
            
            logging.info(f'\t\t5/5: Creating AUROC and AUPRC graph')
            
            # Calculate the AUROC and AUPRC
            auroc = grn_stats.calculate_auroc(confusion_matrix_score_dict)
            auprc = grn_stats.calculate_auprc(confusion_matrix_score_dict)
            
            # Plot the AUROC and AUPRC
            
            auc_path = f'{sample_result_dir}/auroc_auprc.png'
            plotting.plot_auroc_auprc(confusion_matrix_dict_all_methods, auc_path)

            total_accuracy_metrics[method]['auroc'].append(auroc)
            total_accuracy_metrics[method]['auprc'].append(auprc)
        
        # Create a dataframe of the total accuracy metrics by each sample for each inference method
        logging.debug(f'\n----- Saving Accuracy Metrics for All Samples -----')
        for method in method_inferred_net_dict.keys():
            total_accuracy_metrics_df = pd.DataFrame(total_accuracy_metrics[method]).T
            random_accuracy_metrics_df = pd.DataFrame(random_accuracy_metrics[method]).T

            total_accuracy_metrics_df.to_csv(f'OUTPUT/total_accuracy_metrics.tsv', sep='\t')
            random_accuracy_metrics_df.to_csv(f'OUTPUT/random_accuracy_metrics.tsv', sep='\t')
        
        logging.debug(f'\nDone!')
    
    # logging.info(f'\n----- Resource Analysis -----')
    # for method in method_names:
    #     logging.info(f'\tAnalysing resource requirements for {method}')
    #     log_dir = path_dict[method]["resource_log_dir"]
    #     output_dir = path_dict[method]["resource_analysis_dir"]
        
    #     resource_dict = resource_analysis.parse_time_module_output(log_dir, samples)
        
    #     plot_resources_by_step(resource_dict, output_dir)
    #     plot_resources_by_sample(resource_dict, output_dir)
    #     create_resource_requirement_summary(resource_dict, output_dir) 
    
    #     logging.info(f'\nDone! Results written to {output_dir}')

if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')   
    
    main()
    