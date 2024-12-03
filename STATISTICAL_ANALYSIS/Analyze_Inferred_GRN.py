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
        
    print(f'\nInferred network files:')
    for method, sample_dict in inferred_network_dict.items():
        print(f'\tInferred network {method}')
        for sample_name, inferred_network_path in sample_dict.items():
            print(f'\t\t{sample_name} = {inferred_network_path}')
    
    print(f'\nGround truth file:')
    print(f'\t{ground_truth_path}')
    
    return ground_truth_path, method_names, sample_names, inferred_network_dict

def load_inferred_network_df(inferred_network_file, separator):
    return pd.read_csv(inferred_network_file, sep=separator, index_col=0, header=0)

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

def write_method_accuracy_metric_file(total_accuracy_metric_dict):
    for method in total_accuracy_metric_dict.keys():
        total_accuracy_metrics_df = pd.DataFrame(total_accuracy_metric_dict[method]).T
        total_accuracy_metrics_df.to_csv(f'OUTPUT/{method.lower()}_total_accuracy_metrics.tsv', sep='\t')

def preprocess_inferred_and_ground_truth_networks(ground_truth_path, method_names, sample_names, inferred_network_dict):
    # Read in and standardize the ground truth dataframe
    print(f'Reading ground truth')
    ground_truth = pd.read_csv(ground_truth_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = standardize_ground_truth_format(ground_truth)
    ground_truth_dict = create_ground_truth_copies(ground_truth, method_names, sample_names, inferred_network_dict)
    
    processed_inferred_network_dict: dict = {}
    processed_ground_truth_dict: dict = {}
    
    # Format and process the ground truth and inferred networks for each sample in each method
    for method in method_names:
        print(f'\nProcessing {method}')
        method_samples = [sample for sample in sample_names if sample in inferred_network_dict[method]]
        for sample in method_samples:
            print(f'\n\tProcessing {sample}')
            sample_ground_truth = ground_truth_dict[method][sample]
            inferred_network_file = inferred_network_dict[method][sample]
            
            if method == 'CELL_ORACLE':
                sep = ','
            else:
                sep = '\t'
            
            print(f'\t\tLoading inferred network df')
            inferred_network_df = load_inferred_network_df(inferred_network_file, sep)
            
            print(f'\t\tStandardizing dataframe format')
            if method == "CELL_ORACLE":
                inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df, source_col='source', target_col='target', score_col='coef_abs')
            else:
                inferred_network_df = grn_formatting.create_standard_dataframe(inferred_network_df, source_col='Source', target_col='Target', score_col='Score')

            print(f'\t\tAdding inferred scores to ground truth')
            sample_ground_truth = grn_formatting.add_inferred_scores_to_ground_truth(sample_ground_truth, inferred_network_df)
            
            print(f'\t\tSetting scores to log2')
            inferred_network_df["Score"] = np.log2(inferred_network_df["Score"])
            sample_ground_truth["Score"] = np.log2(sample_ground_truth["Score"])
            
            print(f'\t\tRemoving genes from inferred not in ground truth')
            inferred_network_df = grn_formatting.remove_tf_tg_not_in_ground_truth(sample_ground_truth, inferred_network_df)
            
            print(f'\t\tRemoving ground truth edges from inferred')
            inferred_network_df = grn_formatting.remove_ground_truth_edges_from_inferred(sample_ground_truth, inferred_network_df)
            
            print(f'\t\tRemoving NaN values from the inferred network')
            inferred_network_df = inferred_network_df.dropna(subset=['Score'])
            sample_ground_truth = sample_ground_truth.dropna(subset=['Score']) 
            
            print(f'\t\tClassifying interactions by ground truth threshold')
            sample_ground_truth, inferred_network_df = grn_stats.classify_interactions_by_threshold(sample_ground_truth, inferred_network_df)
            
            print(f'\t\tAdding the processed inferred network to the processed inferred network dict')
            if method not in processed_inferred_network_dict:
                processed_inferred_network_dict[method] = {}
            processed_inferred_network_dict[method][sample] = inferred_network_df
            
            print(f'\t\tAdding the processed ground truth dict to the processed ground truth dict')
            if method not in processed_ground_truth_dict:
                processed_ground_truth_dict[method] = {}
            processed_ground_truth_dict[method][sample] = sample_ground_truth
    
    return processed_ground_truth_dict, processed_inferred_network_dict


def main():
    print(f'Reading input files')
    ground_truth_path, method_names, sample_names, inferred_network_dict = read_input_files()
    
    print(f'Preprocessing inferred and ground truth networks')
    processed_ground_truth_dict, processed_inferred_network_dict = preprocess_inferred_and_ground_truth_networks(ground_truth_path, method_names, sample_names, inferred_network_dict)
    
    total_method_confusion_scores = {}
    total_accuracy_metrics = {}
    random_accuracy_metrics = {}
    for method, sample_dict in processed_inferred_network_dict.items():
        print(method)
        total_method_confusion_scores[method] = {'y_true':[], 'y_scores':[]}
        randomized_method_dict = {
                f"{method} Original": {'y_true':[], 'y_scores':[]},
                f"{method} Randomized": {'y_true':[], 'y_scores':[]}
            }
        total_accuracy_metrics[method] = {}
        random_accuracy_metrics[method] = {}
        for sample in sample_dict:
            print(f'\t{sample}')
            
            total_accuracy_metrics[method][sample] = {}
            random_accuracy_metrics[method][sample] = {}

            processed_ground_truth_df = processed_ground_truth_dict[method][sample]
            processed_inferred_network_df = processed_inferred_network_dict[method][sample]
    
            accuracy_metric_dict, confusion_matrix_score_dict = grn_stats.calculate_accuracy_metrics(processed_ground_truth_df, processed_inferred_network_df)
            randomized_accuracy_metric_dict, randomized_confusion_matrix_dict = grn_stats.create_randomized_inference_scores(processed_ground_truth_df, processed_inferred_network_df)

            # Write out the accuracy metrics to a tsv file
            with open(f'./OUTPUT/{method.upper()}/{sample.upper()}/accuracy_metrics.tsv', 'w') as accuracy_metric_file:
                accuracy_metric_file.write(f'Metric\tScore\n')
                for metric_name, score in accuracy_metric_dict.items():
                    accuracy_metric_file.write(f'{metric_name}\t{score:.4f}\n')
                    total_accuracy_metrics[method][sample][metric_name] = score
            
            # Write out the randomized accuracy metrics to a tsv file
            with open(f'./OUTPUT/{method.upper()}/{sample.upper()}/randomized_accuracy_method.tsv', 'w') as random_accuracy_file:
                random_accuracy_file.write(f'Metric\tOriginal Score\tRandomized Score\n')
                for metric_name, score in accuracy_metric_dict.items():
                    random_accuracy_file.write(f'{metric_name}\t{score:.4f}\t{randomized_accuracy_metric_dict[metric_name]:4f}\n')
                    random_accuracy_metrics[method][sample][metric_name] = randomized_accuracy_metric_dict[metric_name]
            
            # Calculate the AUROC and randomized AUROC
            auroc = grn_stats.calculate_auroc(confusion_matrix_score_dict)
            randomized_auroc = grn_stats.calculate_auroc(randomized_confusion_matrix_dict)
            print(f'\t\tAUROC = {auroc:.2f} (randomized = {randomized_auroc:.2f})')
            
            # Calculate the AUPRC and randomized AUPRC
            auprc = grn_stats.calculate_auprc(confusion_matrix_score_dict)
            randomized_auprc = grn_stats.calculate_auprc(randomized_confusion_matrix_dict)
            print(f'\t\tAUPRC = {auprc:.2f} (randomized = {randomized_auroc:.2f})')
            
            # Plot the normal and randomized AUROC and AUPRC for the individual sample
            confusion_matrix_with_method = {method: confusion_matrix_score_dict}
            randomized_confusion_matrix_with_method = {method: randomized_confusion_matrix_dict}
            plotting.plot_auroc_auprc(confusion_matrix_with_method, f'./OUTPUT/{method}/{sample}/auroc_auprc.png')
            plotting.plot_auroc_auprc(randomized_confusion_matrix_with_method, f'./OUTPUT/{method}/{sample}/randomized_auroc_auprc.png')
            
            # Record the y_true and y_scores for the current sample to plot all sample AUROC and AUPRC between methods
            total_method_confusion_scores[method]['y_true'].append(confusion_matrix_score_dict['y_true'])
            total_method_confusion_scores[method]['y_scores'].append(confusion_matrix_score_dict['y_scores'])
            
            # Record the original and randomized y_true and y_scores for each sample to compare against the randomized scores
            randomized_method_dict[f"{method} Original"]['y_true'].append(confusion_matrix_score_dict['y_true'])
            randomized_method_dict[f"{method} Original"]['y_scores'].append(confusion_matrix_score_dict['y_scores'])
            
            randomized_method_dict[f"{method} Randomized"]['y_true'].append(randomized_confusion_matrix_dict['y_true'])
            randomized_method_dict[f"{method} Randomized"]['y_scores'].append(randomized_confusion_matrix_dict['y_scores'])
            
            
            print(f'\tAppending accuracy metrics to total accuracy metric dict')
            
            # Add the auroc and auprc values to the total accuracy metric dictionaries
            total_accuracy_metrics[method][sample]['auroc'] = auroc
            total_accuracy_metrics[method][sample]['auprc'] = auprc
            total_accuracy_metrics[method][sample]['true_positive'] = int(confusion_matrix_score_dict["true_positive"])
            total_accuracy_metrics[method][sample]['true_negative'] = int(confusion_matrix_score_dict["true_negative"])
            total_accuracy_metrics[method][sample]['false_positive'] = int(confusion_matrix_score_dict["false_positive"])
            total_accuracy_metrics[method][sample]['false_negative'] = int(confusion_matrix_score_dict["false_negative"])
            
            random_accuracy_metrics[method][sample]['auroc'] = randomized_auroc
            random_accuracy_metrics[method][sample]['auprc'] = randomized_auprc
            random_accuracy_metrics[method][sample]['true_positive'] = int(randomized_confusion_matrix_dict["true_positive"])
            random_accuracy_metrics[method][sample]['true_negative'] = int(randomized_confusion_matrix_dict["true_negative"])
            random_accuracy_metrics[method][sample]['false_positive'] = int(randomized_confusion_matrix_dict["false_positive"])
            random_accuracy_metrics[method][sample]['false_negative'] = int(randomized_confusion_matrix_dict["false_negative"])
        
        plotting.plot_multiple_method_auroc_auprc(randomized_method_dict, f'./OUTPUT/{method.lower()}_randomized_auroc_auprc.png')
    
    plotting.plot_multiple_method_auroc_auprc(total_method_confusion_scores, './OUTPUT/auroc_auprc_combined.png')
    
    write_method_accuracy_metric_file(total_accuracy_metrics)
    write_method_accuracy_metric_file(random_accuracy_metrics)
    

    
    for method, sample_dict in total_accuracy_metrics.items():
        print(method)
        for sample_name, accuracy_metric_dict in sample_dict.items():
            print(f'\t{sample_name}')
            for accuracy_metric, score in accuracy_metric_dict.items():
                print(f'\t\t{accuracy_metric} = {score:.2f}')
                
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')  
    
    main()