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
├── notebooks/      # Jupyter notebooks for testing and experimentation.
├── plots/          # Generated plots and visualizations.
├── .gitattributes  # Excludes \*.ipynb files from the repository's language statistics.
├── .gitignore      # Specifies files and folders to exclude from Git version control.
└── README.md       # Project documentation and usage instructions.

```

## Configuration: Pipeline Stages Description

### Stage 1: Data Importing, Cleaning, and Preprocessing

This stage handles the acquisition and initial preparation of the stellar dataset by:

1. **Importing Data**  
   - Loads a local CSV file from the `/data` directory.

2. **Imputing Missing Values**  
   - For each column, any missing values (NaN) are replaced with the mean of that column. This helps preserve the overall statistical properties of the data while preventing errors in subsequent modeling stages.

3. **Saving Cleaned Data**  
   - The cleaned data is saved as `cleaned_data.csv` in the `/data` directory, ready for the next stages of the pipeline.

**Input:** A direct query to the Gaia DR3 database or raw CSV file (e.g., `dataset.csv`) in the `/data` folder.

**Output:** `cleaned_data.csv` in the `/data` folder.

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