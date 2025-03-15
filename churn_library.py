"""
Library for the udacity project "Predict customer churn with clean code".

Description:
The churn_library.py is a library of functions to find customers who are likely to churn.
You may be able to complete this project by completing each of these functions,
but you also have the flexibility to change or add functions to meet the rubric criteria.

Author: Dmytro Kysylychyn
Creation date: 15.03.2025

Clean code runs:
        pylint churn_library.py
        autopep8 --in-place --aggressive --aggressive churn_library.py
"""

# import libraries
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
# if pylint gives an error - ensure that pylint and dependencies are up to date
#       'pip install --upgrade pylint astroid'
import seaborn as sns
sns.set_theme()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',  # a - append, w - write
    datefmt="%Y-%m-%d %H:%M:%S",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        assert isinstance(pth, str)
        df = pd.read_csv(pth)
        logging.info("SUCCESS: File '%s' is loaded with %d rows.",
                     pth, df.shape[0])
        return df

    except AssertionError:
        logging.error("File path '%s' is not string.", pth)
        return None

    except FileNotFoundError:
        logging.error("File '%s' is not found.", pth)
        return None


def save_plot(attr_name, plot_pth):
    """
    plot and save figure of attribute to path
    input:
            attr_name: (char)
            plot_pth: (str)

    output:
            None
    """
    try:
        assert isinstance(attr_name, str)
        assert isinstance(plot_pth, str)
        plt.title(attr_name)
        plt.savefig(plot_pth + "/" + attr_name + ".png")
        logging.info(
            "SUCCESS: Figure '%s' saved to '%s'.",
            attr_name,
            plot_pth)
        return None

    except AssertionError:
        logging.error(
            "Cannot save '%s': path '%s' is not string.",
            attr_name,
            plot_pth)
        return None

    except FileNotFoundError:
        logging.error(
            "Cannot save '%s': path '%s' is not found.",
            attr_name,
            plot_pth)
        return None


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder: 'images/eda'
    input:
            df: pandas dataframe

    output:
            None
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio']

    print(df.head())

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # create and save histograms to 'images/eda' folder
    # Churn
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    save_plot("Churn", "images/eda")

    # Customer_Age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    save_plot("Customer_Age", "images/eda")

    # Marital_Status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    save_plot("Marital_Status", "images/eda")

    # Total_Trans_Ct
    # Show distributions of 'Total_Trans_Ct' and
    # add a smooth curve obtained using a kernel density estimate
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    save_plot("Total_Trans_Ct", "images/eda")

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.select_dtypes(include=[float, int]).corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    save_plot("Heatmap", "images/eda")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    # import_data(2)
    # import_data("data/bank_data1.csv")
    df = import_data("data/bank_data.csv")
    # print(df)
    perform_eda(df)
