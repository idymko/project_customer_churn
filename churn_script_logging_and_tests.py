"""

Contains unit tests for the churn_library.py functions for each input function.
Uses the basic assert statements that test functions work properly.
The goal of test functions is to checking the returned items aren't empty or
folders where results should land have results after the function has been run.

Logs any errors and INFO messages in a .log file,
so it can be viewed post the run of the script.

Clean code runs:
		pylint churn_script_logging_and_tests.py
		autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
"""
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',  # a - append, w - write
    datefmt="%Y-%m-%d %H:%M:%S",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import
    '''
    try:
        df = cl.import_data("./data/bank_data.csv")
        logging.info("Testing import_data data file: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data dataframe shape: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_save_plot():
    """
    test save_plot
    """
    try:
        df = cl.import_data("./data/bank_data.csv")
        plt.figure(figsize=(20, 10))
        df['Dependent_count'].hist()

        cl.save_plot("Dependent_count", "images/eda")
        assert os.path.isfile("images/eda/Dependent_count.png")
        logging.info("Testing save_plot : SUCCESS")
        os.remove("images/eda/Dependent_count.png")
    except AssertionError:
        logging.error(
            "Testing save_plot: Test plot image not found in 'images/eda/' after function run")


def test_perform_eda():
    '''
    test perform eda function
    '''
    try:
        df = cl.import_data("./data/bank_data.csv")
        cl.perform_eda(df)
        logging.info("Testing perform_eda image location: SUCCESS")
    except FileNotFoundError:
        logging.error(
            "Testing perform_eda: Location 'images/eda' wasn't found")
    except KeyError as e:
        logging.error(
            "Testing perform_eda: Dataframe does not contain a required column name '%s'",
            e)

    try:
        target_files = ['Customer_Age.png', 'Churn.png',
                        'Marital_Status.png', 'Total_Trans_Ct.png',
                        'Heatmap.png']
        for file_ in target_files:
            assert os.path.isfile("images/eda/" + file_)
        logging.info("Testing perform_eda output figures: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing perform_eda: File '%s' not found in 'images/eda/' after function run",
            file_)


def test_encoder_helper():
    '''
    test encoder helper
    '''
    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    response = "Churn"
    try:
        df = cl.import_data("./data/bank_data.csv")
        df = cl.encoder_helper(df, response)
        logging.info("Testing encoder_helper run: SUCCESS")
    except KeyError as e:
        logging.error(
            "Testing encoder_helper run: Dataframe does not contain a required column name %s",
            e)

    try:
        df_columns = df.columns
        for cat_ in category_lst:
            assert cat_ + "_" + response in df_columns
        logging.info("Testing encoder_helper output: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing encoder_helper output: Column '%s' not found in dataframe after function run",
            cat_ + "_" + response)


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    try:
        df = cl.import_data("./data/bank_data.csv")
        df = cl.encoder_helper(df)
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering run: SUCCESS")
    except KeyError as e:
        logging.error(
            "Testing perform_feature_engineering run: Column %s of Dataframe", e)

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering output: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering output: Training or test split data is empty")


def test_save_text_as_image():
    """
    test save_text_as_image
    """
    try:
        cl.save_text_as_image("samlpe text", "images/results/test.png")
        assert os.path.isfile("images/results/test.png")
        logging.info("Testing save_text_as_image : SUCCESS")
        os.remove("images/results/test.png")
    except AssertionError:
        logging.error(
                "Testing save_text_as_image: \
				Test text image not found in \
				'images/results/' after function run")


def test_feature_importance_plot():
    """
    test feature_importance_plot
    """
    try:
        # Sample dataset
        x_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'feature3': [2, 3, 4, 5, 6]
        })
        y_data = [0, 1, 0, 1, 0]  # Target variable

        # Train Random Forest
        model = RandomForestClassifier()
        model.fit(x_data, y_data)

        cl.feature_importance_plot(model, x_data, "images/results")
        assert os.path.isfile(
            "images/results/random_forests_feature_importance.png")
        logging.info("Testing feature_importance_plot output: SUCCESS")
    except AssertionError:
        logging.error(
                "Testing feature_importance_plot: \
				File 'test_feature_importance' \
				not found after function run")


def test_train_models():
    '''
    test train_models
    '''
    try:
        dataframe = cl.import_data("data/bank_data.csv")
        # slide a dataframe for fast testing purposes
        dataframe = dataframe[:100]

        cl.perform_eda(dataframe)

        dataframe = cl.encoder_helper(dataframe)

        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            dataframe)
        cl.train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models run: SUCCESS")
    except KeyError:
        logging.error("Testing train_models run failed")

    try:
        output_files = ["images/results/random_forests_report.png",
                        "images/results/logistic_regression_report.png",
                        "models/rfc_model.pkl", "models/logistic_model.pkl"]
        for file_ in output_files:
            assert os.path.isfile(file_)
        logging.info("Testing train_models output: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing train_models: File '%s' not found after function run",
            file_)


def test_classification_report_image():
    """
    test classification_report_image
    """
    try:
        # Sample ground truth (actual labels) and predictions
        y_true = [0, 1, 1, 2, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 1]

        cl.classification_report_image(y_true, y_true, y_pred, y_pred,
                                       "test_class_report.png")
        assert os.path.isfile("images/results/test_class_report.png")
        logging.info("Testing classification_report_image run: SUCCESS")
        os.remove("images/results/test_class_report.png")
    except AssertionError:
        logging.error(
                "Testing classification_report_image: \
				Test classification report image not \
				found in 'images/results/' after function run")


def test_roc_plot():
    """
    test roc_plot
    """
    try:
        dataframe = cl.import_data("data/bank_data.csv")
        # slide a dataframe for fast testing purposes
        dataframe = dataframe[:100]

        cl.perform_eda(dataframe)

        dataframe = cl.encoder_helper(dataframe)

        _, x_test, _, y_test = cl.perform_feature_engineering(
            dataframe)
        rfc_model = joblib.load('./models/rfc_model.pkl')
        lr_model = joblib.load('./models/logistic_model.pkl')

        cl.roc_plot(rfc_model, lr_model, x_test, y_test, "images/results")
        assert os.path.isfile("images/results/roc_curve_result.png")
        logging.info("Testing roc_plot run: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing roc_plot: \
				roc_curve_result.png image not \
				found in 'images/results/' after function run")


if __name__ == "__main__":
    # test all functions from 'churn_library'
    test_import()
    test_save_plot()
    test_perform_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
    test_save_text_as_image()
    test_classification_report_image()
    test_feature_importance_plot()
    test_roc_plot()
