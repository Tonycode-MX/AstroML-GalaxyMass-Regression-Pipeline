# Main module for executing stages

# LIBRARIES

# Standard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# External modules
from modules.module_data_path import data_path, plot_data_path
from modules.module_data_cleaning import nans_elimination
from modules.module_utils import add_color_magnitude_indices, label_star, star_counts, save_dataframe, save_list_to_file, load_list_from_file

# Number of stages to execute (1 to 5)
stage = [1]  # Change this list to execute different stages

print("\nExecuting Stage", stage[0])

#Stage 1: Data Loading and Preprocessing
def stage1():
    # Get data path
    data_folder = data_path()

    #Dataset name
    data_name = "Buzzard_DC1" # Change this to the desired dataset name without extension (e.g., "Buzzard_DC1" for "Buzzard_DC1.csv")

    # Load dataset
    dataset = os.path.join(data_folder, data_name + ".csv")
    df = pd.read_csv(dataset)

    # Data Cleaning
    df_cleaned = nans_elimination(df)

    # Save the imported DataFrame as CSV in /data
    save_dataframe(df_cleaned, data_folder, filename= data_name + "_cleaned.csv")

    # Preview (optional)
    #print("\nImported dataset preview:")
    #print(df_cleaned.head())

def stage2():
    print("Executing Stage 2")
    # Add stage 2 specific logic here

def stage3():
    print("Executing Stage 3")
    # Add stage 3 specific logic here

def stage4():
    print("Executing Stage 4")
    # Add stage 4 specific logic here

def stage5():
    print("Executing Stage 5")
    # Add stage 5 specific logic here

# Execute stages
if __name__ == '__main__': 
    if 1 in stage:
        stage1()
    elif 2 in stage:
        stage2()
    elif 3 in stage:
        stage3()
    elif 4 in stage:
        stage4()
    elif 5 in stage:
        stage5()