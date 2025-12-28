# AstroML-GalaxyMass-Regression-Pipeline


This project was developed for Partial 2 of the Deep Learning course as part of my academic exchange at the Universidad Católica de Colombia (Bogotá, Colombia). ...

**Note on Methodology:**

Description

### Models Used

This project includes three supervised machine learning models for stellar classification, each with different characteristics:

1. **Model 1**  
   - Description

2. **Model 2**  
   - Description

3. **Model 3**  
   - Description

By evaluating these three models under the same pipeline, the project compares their ... (metrics)

## Additional Options (Other Strategies)

Description
This is an example:

```
+--------------+-----------------+---------------+-----------------------------------------------------+
|   Model Key  | Base Classifier |    Strategy   |                       Purpose                       |
+--------------+-----------------+---------------+-----------------------------------------------------+
|   KNN_OVR    |       KNN       |  One-vs-Rest  | Applies KNN in a binary fashion for                 |
|              |                 |               | each class, potentially improving class separation. |
+--------------+-----------------+---------------+-----------------------------------------------------+
|   RF_OVR     |  Random Forest  |  One-vs-Rest  | Uses RF's robustness within the OVR framework.      |
+--------------+-----------------+---------------+-----------------------------------------------------+
|              |                 |               | Leverages SVM's precision in a multi-class scenario |
|   SVM_OVR    |       SVM       |  One-vs-Rest  | by training an independent classifier for each      |
|              |                 |               | stellar population.                                 |
+--------------+-----------------+---------------+-----------------------------------------------------+
```

## Repository Structure
```
.
├── data/           # Original data (.csv).
├── environment/    # Conda environment configuration files.
├── modules/        # Python modules and scripts. Main entry point: main.py
├── plots/          # Generated plots and visualizations.
├── .gitattributes  # Excludes \*.ipynb files from the repository's language statistics.
├── .gitignore      # Specifies files and folders to exclude from Git version control.
└── README.md       # Project documentation and usage instructions.

```

## Configuration: Pipeline Stages Description

### Stage 1: Data Loading and Preprocessing.

This stage handles the acquisition and initial preparation of the stellar dataset by:

1. **Importing Data**  
   - Loads a local CSV file from the `/data` directory.

2. **Imputing Missing Values**  
   - For each column, any missing values (NaN) are replaced with the mean of that column. This helps preserve the overall statistical properties of the data while preventing errors in subsequent modeling stages.

3. **Saving Cleaned Data**  
   - The cleaned data is saved as `data_name` + `_cleaned.csv` in the `/data` directory, ready for the next stages of the pipeline. 
     (`data_name` is the desired dataset name without extension)


**Input:** A direct query to the Gaia DR3 database or raw CSV file (e.g., `Buzzard_DC1.csv`) in the `/data` folder.

**Output:** `data_name_cleaned.csv` in the `/data` folder.

### Stage 2: Feature Engineering & Labeling

This stage focuses on the exploratory data analysis (EDA) and visualization of the processed dataset to understand feature distributions and relationships by:

1. **Loading Processed Data**

   - Loads the cleaned dataset (`data_name` + `_cleaned.csv`) generated in the previous stage from the `/data` directory.

2. **Generating Exploratory Visualizations**

   - Scatter Plots: Creates plots to analyze the relationship between individual features and the target variable.

   - Boxplots: Generates boxplots to visualize statistical distributions and identify potential outliers within the features.

   - Correlation Heatmap: Constructs a heatmap to visualize the correlation matrix, highlighting dependencies between variables.

3. **Saving Visualizations**

   - The generated figures are saved as high-quality PDF files (`scatter_plots.pdf`, `boxplots.pdf`, and `correlation_heatmap.pdf`) in the `/plots` directory for review and reporting.

**Input:** The cleaned CSV file (e.g., `Buzzard_DC1.csv`) located in the `/data` folder.

**Output:** PDF visualization files (`scatter_plots.pdf`, `boxplots.pdf`, and `correlation_heatmap.pdf`) in the `/plots` folder.

### Stage 3: RF & RFE-based Modeling

This stage manages the preparation of data for machine learning and performs feature selection to identify the most relevant variables by:

1. **Preparing and Splitting Data**
   - Splits the cleaned dataset into training, validation, and testing sets (Train/Val/Test) and performs necessary encoding.Saves these subsets as individual CSV files (e.g., `X_train.csv`, `y_train.csv`, etc.) in the `/data` directory for consistent usage across models.

2. **Selecting Features**
   - Identifies the top `K` most significant features (e.g., `N=5`) using two distinct methods: Random Forest feature importance and Recursive Feature Elimination (RFE).

3. **Benchmarking Performance**
   - Evaluates and compares the model's performance when trained on the full set of features versus the reduced subset of selected features, validating results against the validation set.

4. **Saving Feature Artifacts**
   - The lists of optimal features identified by both methods are serialized and saved as .pkl files (`topK_features_rfe.pkl` and `topK_features_rf.pkl`) in the `/data` directory.

**Input:** `data_name_cleaned.csv` in the `/data` folder.

**Output:** Train/Val/Test split CSV files (`X_train.csv`, `X_val.csv`, `X_test.csv`, etc.) and feature list pickle files (`topK_features_rfe.pkl`, `topK_features_rf.pkl`) in the `/data` folder.

### Stage 4: Model Comparison & Selection

This stage evaluates performance across different algorithms to identify the optimal model for the specific task by:

1. **Loading Data & Features**

   - Imports the list of optimal features (`topK_features_rfe.pkl`) selected in the previous stage.

   - Loads the training and validation datasets (`X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`) from the `/data` directory, filtering the feature matrices to include only the columns corresponding to the selected features.

2. **Benchmarking Models**

   - Instantiates a set of candidate algorithms (e.g., `Linear Regression`, `Ridge, Lasso`, `Random Forest`) and evaluates their performance on the validation set.

   - Generates a detailed report to compare metrics and determines the best-performing estimator.

3. **Saving Best Model Configuration**

   - The name of the model with the highest performance is identified and saved as a pickle file (`best_model_name.pkl`) in the `/data` directory, marking it for final tuning and testing.

**Input:** Split data files (`X_train.csv`, `X_val.csv`, `y_train.csv`, `y_val.csv`) and the feature list (`topK_features_rfe.pkl`) in the `/data` folder.

**Output:** `best_model_name.pkl` in the `/data` folder.

### Stage 5: Final Model Training & Evaluation

This stage executes the final production-ready training and assessment of the selected machine learning model by:

1. **Retrieving Artifacts**

   - Imports the optimal feature list (`topK_features_rfe.pkl`) and the best-performing model architecture (`best_model_name.pkl`) identified in previous stages.

2. **Loading & Filtering Data**

   - Loads the Training, Validation, and Test datasets from the `/data` directory, filtering the dataframes to retain only the columns corresponding to the selected features.

3. **Retraining & Final Evaluation**

   - Retrains the best model instance on the combined Training and Validation sets to maximize the learning data.

   - Evaluates the final model on the held-out Test set to provide an unbiased estimate of performance.

4. **Saving Final Model**

   - The fully trained model is serialized and saved as a .joblib file (e.g., `ModelName_final_model.joblib`) in the `/data` directory, ready for inference or deployment.

**Input:** Split data files (`X_train/val/test.csv`, `y_train/val/test.csv`), feature list (`topK_features_rfe.pkl`), and model name (`best_model_name.pkl`) in the `/data` folder.

**Output:** The serialized model file (e.g., `RandomForest_final_model.joblib`) in the `/data` folder.

## Conda environment setup

Inside directory `environment/` there is a file named `astrophysics.yml`. This file is used to set up a dedicated Conda environment with all the necessary dependencies for running the code in this repository.

To create the environment, first ensure you have **Anaconda** or **Miniconda** installed on your system. You can download it from [Anaconda's official website](https://www.anaconda.com/download). Then, open a terminal and run the following command:


```bash
conda env create -f astrophysics.yml
```

This command will create a new Conda environment named `astrophysics`, installing all required libraries and dependencies automatically.

#### Activating and Deactivating the Environment

Once the installation is complete, you can activate the new environment by running:


```bash
conda activate astrophysics
```

If you need to switch to another environment or deactivate it, simpy run:

```bash
conda deactivate
```

## File Format for CSV Files

Description
```
+-----------+-----------+-----------+-----------+-----------+---------+
| Variable1 | Variable2 | Variable3 |    ...    | VariableK | Target  |
+-----------+-----------+-----------+-----------+-----------+---------+
|   0.82    |   4.76    |   7.13    |    ...    |   1.01    |   MS    |
|   1.35    |  -1.20    |   3.25    |    ...    |   1.02    |  Giant  |
|  -0.10    |  11.05    |  12.80    |    ...    |   1.03    |   WD    |
+-----------+-----------+-----------+-----------+-----------+---------+
```

## Running the Main Script

The `modules/` directory contains the main script **`main.py`**, which serves as the primary entry point for executing the analysis process. To run the script, you must first navigate to the ‘modules/‘ directory. Use the following commands in your terminal (with your Conda environment already activated):
   - 1. Navigate to the modules directory
```bash
cd modules/
```
   - 2. Run the main script
```bash
python main.py
```

#### Configuring Execution Stages

At the beginning of `main.py`, there is a specific line that defines which stages of the analysis process will be executed:

```python
stages = [1] will only run Stage 1.
stages = [2] will only run Stage 2.
stages = [3] will only run Stage 3.
stages = [4] will only run Stage 4.
stages = [5] will only run Stage 5.
```

#### Acknowledgements
I would like to express my special thanks to:
   - Alondra Plascencia (Universidad de Guadalajara, [@Alondra-Plascencia](https://github.com/Alondra-Plascencia)), for her unwavering support during this academic exchange. Her continuous encouragement and assistance were fundamental in allowing us to successfully complete projects like this one, significantly contributing to our professional growth.