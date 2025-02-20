import os
import copy
import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse

# Import necessary modules from linger
sys.path.insert(0, '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER')

import BASH_LINGER.shared_variables as shared_variables

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

# The path to the LOGS directory
LOG_DIR = '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/LOGS/'

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

if __name__ == '__main__':
    # Define a dictionary to hold the sample names with their resource requirements for each step in the pipeline
    sample_resource_dict = {}
    
    samples = [i for i in os.listdir(LOG_DIR)]
    
    # samples = ["K562_human_filtered", "1000_cells_E7.5_rep1"]
    
    sample_list = [
        sample_dir for sample_dir in os.listdir(LOG_DIR)
        if sample_dir in samples
        # if rep in sample_dir for rep in samples
    ]

    for sample_log_dir in os.listdir(LOG_DIR):
        
        
        # Find each sample in the LOGS directory
        if sample_log_dir in sample_list:
            # print(f'Analyzing {sample_log_dir}')
            
            # Initialize pipeline_step_dict once per sample_log_dir
            sample_resource_dict[sample_log_dir] = {}
            
            # Find each step log file for the sample
            for file in os.listdir(f'{LOG_DIR}/{sample_log_dir}'):
                
                if file.endswith(".log"):
                    
                    steps = [f'Step_0{i}0' for i in [1, 2, 3, 4]]
                    pipeline_step = file.split(".")[0]
                    
                    if pipeline_step in steps:
                        # print(f'\t{pipeline_step}')
                    
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
                                if "Command exited with non-zero status 1" in line:
                                    for _ in range(22):
                                        next(log_file)
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
                            
    # If the user time is 0, it means that the step failed
    failed_samples = set()
    for sample_log_dir in sample_resource_dict:
        print(sample_log_dir)
        for pipeline_step in sample_resource_dict[sample_log_dir]:
            print(pipeline_step)
            if sample_resource_dict[sample_log_dir][pipeline_step]["user_time"] == 0:
                failed_samples.add(sample_log_dir)
    
    for sample in failed_samples:
        print(f'{sample} did not complete a step, removing from time consideration')
        sample_resource_dict.pop(sample)
                                

    summary_dict = {}

    for sample, step_dict in sample_resource_dict.items():
        # print(sample)
        if sample not in summary_dict:
            summary_dict[sample] = {
                    "user_time": 0,
                    "system_time": 0,
                    "percent_cpu": [],
                    "wall_clock_time": 0,
                    "max_ram": []
                }
        for step, resource_dict in step_dict.items():
            # print(f'\t{step}')
            for resource_name, resource_value in resource_dict.items():
                # print(f'\t\t{resource_name}: {resource_value}')
                if resource_name == "percent_cpu":
                    summary_dict[sample][resource_name].append(round(resource_value,2))
                elif resource_name == "max_ram":
                    summary_dict[sample][resource_name].append(round(resource_value,2))
                else:
                    summary_dict[sample][resource_name] += round(resource_value,2)
                    
        summary_dict[sample]["max_ram"] = max(summary_dict[sample]["max_ram"])
        summary_dict[sample]["percent_cpu"] = round(sum(summary_dict[sample]["percent_cpu"]) / len(summary_dict[sample]["percent_cpu"]),2)
        
    summary_df = pd.DataFrame(summary_dict)
    # summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)
    print(summary_df.head())
    
    if not os.path.exists("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/RESOURCE_ANALYSIS/"):
        os.makedirs("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/RESOURCE_ANALYSIS/")

    summary_df.to_csv(f'/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/BASH_LINGER/RESOURCE_ANALYSIS/Resource_Summary.tsv', sep='\t')