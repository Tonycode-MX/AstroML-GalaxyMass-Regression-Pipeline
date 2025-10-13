import numpy as np
import pandas as pd
import pickle
import os

def save_dataframe(df, data_folder, filename="dataframe.csv"):
    """
    Saves the DataFrame to a CSV file at the specified path.
    Args:
        df (pd.DataFrame): DataFrame to save.
        data_folder (str): Path to the folder where the file will be saved.
        filename (str): Name of the output file (default is 'dataframe.csv').
    Returns:
        None
    """
    save_path = os.path.join(data_folder, filename)
    df.to_csv(save_path, index=False)
    print(f"\nDataFrame saved to {save_path}")

def add_color_magnitude_indices(df):
    """
    Adds color and magnitude indices to the DataFrame.

    Color index: BP_RP = phot_bp_mean_mag - phot_rp_mean_mag
    Magnitude index: M_G = phot_g_mean_mag + 5 - 5*log10(d_pc), where d_pc = 1000/parallax

    Args:
        df (pd.DataFrame): DataFrame containing the necessary columns.
    Returns:
        pd.DataFrame: DataFrame with added 'BP_RP' and 'M_G' columns.
    """
    df["BP_RP"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    d_pc = 1000.0 / df["parallax"]   # distancia en parsecs (parallax en mas)
    df["M_G"] = df["phot_g_mean_mag"] + 5 - 5*np.log10(d_pc)
    return df

def label_star(row):
    """
    Labels a star based on its color and magnitude indices.
    Criteria:
      - White Dwarf (WD): -0.5 <= BP_RP <= 1.8 and M_G >= 10
      - Giant: BP_RP >= 0.8 and M_G <= 2.5
      - Main Sequence (MS): all other cases
    Args:
        row (pd.Series): A row of the DataFrame containing 'BP_RP' and 'M_G' columns.
    Returns:
        str: The label of the star ("WD", "Giant", or "MS").
    """
    # White Dwarfs: blue and faint
    if (-0.5 <= row.BP_RP <= 1.8) and (row.M_G >= 10):
        return "WD"      # White Dwarf
    # Giants: red and bright
    elif (row.BP_RP >= 0.8) and (row.M_G <= 2.5):
        return "Giant"
    # Main Sequence: intermediate
    else:
        return "MS"      # Main Sequence
    
def star_counts(df):
    """
    Prints the counts of each star type in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing a 'target' column with star labels.
    Returns:
        None
    """
    print("\n=================")
    print("Star type counts:")
    print("=================")
    print(df["target"].value_counts())
    print("=================\n")

def save_list_to_file(list_to_save, folder_path, filename="list.pkl"):
    """
    Saves a list of items to a pickle file, one item per line.
    Args:
        list_to_save (list): List of items to save.
        folder_path (str): Path to the folder where the file will be saved.
        filename (str): Name of the output file (default is 'list.pkl').
    Returns:
        None
    """
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    with open(full_path, 'wb') as file:
        pickle.dump(list_to_save, file)

    print(f"\nList saved as '{filename}' successfully.")


def load_list_from_file(folder_path, filename="list.pkl"):
    """
    Loads a list from a pickle file at a specified path and returns it.
    
    Args:
        folder_path (str): The directory where the file is located.
        filename (str): The name of the pickle file.
        
    Returns:
        list: The list loaded from the file, or None if the file is not found.
    """
    # build full file path
    full_path = os.path.join(folder_path, filename)
    
    # load list from pickle file
    try:
        with open(full_path, 'rb') as file:
            loaded_list = pickle.load(file)
        
        print(f"\nList loaded from '{full_path}' successfully.")
        return loaded_list
    
    except FileNotFoundError:
        print(f"\nError: The file '{full_path}' was not found.")
        return None