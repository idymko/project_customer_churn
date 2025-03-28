# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, credit card customers who are most likely to churn are identified. The completed project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package can also be run interactively or from the command-line interface (CLI). The best coding practices for testing and logging are implemented.

## Files and data description
Overview of the files and data present in the root directory. 
```bash 
    .
    ├── Guide.ipynb          # Getting started and troubleshooting tips
    ├── churn_notebook.ipynb # Contains the code to be refactored
    ├── churn_library.py     # Contains the necessary functions
    ├── churn_script_logging_and_tests.py # Contains tests and logs
    ├── README.md            # Provides project overview, and instructions to use the code
    ├── data                 # Project data file
    │   └── bank_data.csv
    ├── images               
    │   ├── eda              # Location for EDA results
    │   └── results          # Location for reports and model evaluation
    ├── logs                 # Location for logs
    └── models               # Location for models
```

## Running Files

1. Install requirements: `python -m pip install -r requirements_py3.10.txt`
2. Run testing and logging: `python churn_script_logging_and_tests.py`

    Logs any errors and INFO messages in a `./logs/churn_library.log` file,
    so it can be viewed post the run of the script.

3. Run the training model training: `python churn_library.py`
