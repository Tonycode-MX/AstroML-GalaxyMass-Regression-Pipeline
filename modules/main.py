# Main module for executing stages

# LIBRARIES

# Standard
import pandas as pd
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# External modules
from modules.module_data_path import data_path, plot_data_path
from modules.module_data_cleaning import nans_elimination
from modules.module_utils import save_dataframe, save_list_to_file, load_list_from_file, reorder_and_rename_target
from modules.module_utils import plot_feature_vs_target_scatter, plot_boxplots, plot_correlation_heatmap
from modules.module_models import prepare_data_for_modeling, rf_feature_selection, rfe_feature_selection, compare_feature_selection
from modules.module_models import import_models, compute_metrics, compare_in_validation, compare_model_performance, retrain_best_model, save_model

# Number of stages to execute (1 to 5)
stage = [5]  # Change this list to execute different stages

print("\nExecuting Stage", stage[0])

#Dataset name
data_name = "Buzzard_DC1" # Change this to the desired dataset name without extension (e.g., "Buzzard_DC1" for "Buzzard_DC1.csv")

#STAGE 1: Data Loading and Preprocessing
def stage1():
    """
    Data Import and Initial Cleaning Stage.
        1) Defines and retrieves the data path.
        2) Loads the raw dataset from the specified path.
        3) Performs initial data cleaning, specifically handling and eliminating NaN values.
        4) Renames the target variable ('log.mass') to a standardized name ('target') and moves it to the final column position.
        5) Saves the resulting cleaned DataFrame to the '/data' folder for the next stage.
    """
    # Get data path
    data_folder = data_path()

    # Load dataset
    dataset = os.path.join(data_folder, data_name + ".csv")
    df = pd.read_csv(dataset)

    # Data Cleaning
    df_cleaned = nans_elimination(df)

    # Rename target column if necessary and move at the end (optional)
    target_column = 'log.mass'  # Change this to the actual target column name if different
    df_cleaned = reorder_and_rename_target(df_cleaned, old_col_name = target_column, new_col_name= "target")

    # Remove some columns if necessary (optional)
    #columns_to_remove = ['u.err', 'g.err', 'r.err', 'i.err', 'z.err', 'y.err']
    #df_cleaned = df_cleaned.drop(columns=columns_to_remove, errors='ignore')

    # Save the imported DataFrame as CSV in /data
    save_dataframe(df_cleaned, data_folder, filename= data_name + "_cleaned.csv")

    # Preview (optional)
    print("\nImported dataset preview:")
    print(df_cleaned.head())

# STAGE 2: Feature Engineering & Labeling
def stage2():
    """
    Main script for feature engineering and labeling.
        1) Load cleaned data.
        2) Generate EDA plots (scatter plots, boxplots, correlation heatmap).
        3) Save plots to /plots folder.
    """
    # Get data path
    data_folder = data_path()
    plot_folder = plot_data_path()

    dataset = os.path.join(data_folder, data_name + "_cleaned.csv")
    df = pd.read_csv(dataset)

    # EDA PLOTS
    # Scatter plots
    plot_feature_vs_target_scatter(df, save_dir=plot_folder, filename='scatter_plots.pdf')

    # Boxplots
    plot_boxplots(df, save_dir=plot_folder, filename='boxplots.pdf')

    # Correlation heatmap
    plot_correlation_heatmap(df, save_dir=plot_folder, filename='correlation_heatmap.pdf')

# STAGE 3: RF & REF-based Modeling
def stage3():
    """
    Main script for feature selection using Random Forest and RFE.
        1) Prepare data for modeling (feature selection, train-test split, encoding).
        2) Feature selection with Random Forest.
        3) Feature selection with RFE.
        4) Compare feature selection results.
        5) Comparison of training with all features vs. K features and validation.
    """
    # Get data path
    data_folder = data_path()
    dataset = os.path.join(data_folder, data_name + "_cleaned.csv")
    df = pd.read_csv(dataset)

    n_ft = 5  # Number of top features to select

    # Prepare data for modeling
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_modeling(df)

    # Save datasets in /data folder
    save_dataframe(X_train, data_folder, filename="X_train.csv")
    save_dataframe(X_val, data_folder, filename="X_val.csv")
    save_dataframe(X_test, data_folder, filename="X_test.csv")
    save_dataframe(y_train, data_folder, filename="y_train.csv")
    save_dataframe(y_val, data_folder, filename="y_val.csv")
    save_dataframe(y_test, data_folder, filename="y_test.csv")

    # Feature selection with Random Forest
    topK_rf = rf_feature_selection(X_train, y_train, n_features=n_ft, show_importance=False)  #show_importance=True to print importances

    # Feature selection with RFE
    topK_rfe = rfe_feature_selection(X_train, y_train, n_features=n_ft)

    # Compare feature selection results (optional)
    #compare_feature_selection(topK_rf, topK_rfe)

    # Comparison of training with all features vs. K features and validation.
    compare_model_performance(X_train, y_train, X_val, y_val, X_test, y_test, selected_features=topK_rfe)

    # Save top K features to pickle files in /data folder
    save_list_to_file(topK_rfe, data_folder, filename="topK_features_rfe.pkl")
    save_list_to_file(topK_rf, data_folder, filename="topK_features_rf.pkl")

# STAGE 4: Model Comparison & Selection
def stage4():
    """
    Main script for model comparison and selection.
        1) Import selected features from previous stage.
        2) Prepare data for modeling.
        3) Compare models in validation.
        4) Save best model name to pickle file in /data folder.
    """
    # Get data path
    data_folder = data_path()

    #import selected features from previous stage
    best_features = load_list_from_file(data_folder, filename="topK_features_rfe.pkl")    

    # Prepare data for modeling
    X_train = pd.read_csv(os.path.join(data_folder, "X_train.csv"), usecols=best_features)
    X_val   = pd.read_csv(os.path.join(data_folder, "X_val.csv"), usecols=best_features)
    y_train = pd.read_csv(os.path.join(data_folder, "y_train.csv")).squeeze()
    y_val   = pd.read_csv(os.path.join(data_folder, "y_val.csv")).squeeze()

    # Select models to compare 
    # Available models: "LinearRegression", "Ridge", "Lasso", "RandomForestRegressor", "KNNRegressor", "SVR", "DecisionTreeRegressor"
    models = ["LinearRegression", "Ridge", "Lasso", "RandomForestRegressor"]

    # Compare in validation
    best_model = compare_in_validation(import_models(models), X_train, y_train, X_val, y_val, show_detailed_report=True)

    # Save best model name to pickle file in /data folder
    save_list_to_file([best_model], data_folder, filename="best_model_name.pkl")

def stage5():
    """
    Main script for final model training and evaluation.
        1) Import selected features from previous stage.
        2) Prepare data for modeling.
        3) Import best model name from previous stage.
        4) Retrain best model on train+val set.
        5) Save final model to /data folder.
    """
    # Get data path
    data_folder = data_path()

    #import selected features from previous stage
    best_features = load_list_from_file(data_folder, filename="topK_features_rfe.pkl")

    # Prepare data for modeling
    X_train = pd.read_csv(os.path.join(data_folder, "X_train.csv"), usecols=best_features)
    X_val   = pd.read_csv(os.path.join(data_folder, "X_val.csv"), usecols=best_features)
    X_test  = pd.read_csv(os.path.join(data_folder, "X_test.csv"), usecols=best_features)

    y_train = pd.read_csv(os.path.join(data_folder, "y_train.csv")).squeeze()
    y_val   = pd.read_csv(os.path.join(data_folder, "y_val.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(data_folder, "y_test.csv")).squeeze()

    # Import best model name from previous stage
    best_model_name = load_list_from_file(data_folder, filename="best_model_name.pkl")[0]

    # Retrain best model on train+val set
    final_model = retrain_best_model(X_train, y_train, X_val, y_val, X_test, y_test, best_model_name)

    # Save final model to /data folder
    save_model(final_model, data_folder, filename= f"{best_model_name}_" + "final_model.joblib")

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