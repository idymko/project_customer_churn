"""
Library for the udacity project "Predict customer churn with clean code".

Description:
The churn_library.py is a library of functions to find customers who are likely to churn.

Author: Dmytro Kysylychyn
Creation date: 15.03.2025

Clean code runs:
		pylint churn_library.py
		autopep8 --in-place --aggressive --aggressive churn_library.py
"""

# import libraries
import os
import numpy as np
import joblib
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
# if pylint gives an error - ensure that pylint and dependencies are up to date
#       'pip install --upgrade pylint astroid'
import seaborn as sns
sns.set_theme()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
                    pth: a path to the csv
    output:
                    df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df


def save_plot(attr_name, plot_pth):
    """
    plot and save figure of attribute to path
    input:
                    attr_name: (char)
                    plot_pth: (str)

    output:
                    None
    """
    plt.title(attr_name)
    plt.tight_layout()
    plt.savefig(plot_pth + "/" + attr_name + ".png")
    plt.close()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder: 'images/eda'
    input:
                    df: pandas dataframe

    output:
                    None
    '''

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


def encoder_helper(df, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
                    df: pandas dataframe
                    response: string of response name [optional argument that
                                            could be used for naming variables or index y column]

    output:
                    df: pandas dataframe with new columns for
    '''
    # category_lst: list of columns that contain categorical features
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    for _cat in category_lst:
        _lst = []
        _groups = df.groupby(_cat).mean(numeric_only=True)[response]

        for val in df[_cat]:
            _lst.append(_groups.loc[val])

        df[_cat + '_' + response] = _lst

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    Performs training and test split on selected columns
    input:
                      df: pandas dataframe
                      response: string of response name
                                            [optional argument that could be used
                                            for naming variables or index y column]

    output:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    '''
    y = df[response]
    x = pd.DataFrame()
    keep_cols = [
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
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x[keep_cols] = df[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def save_text_as_image(report_str, filename):
    """
    saves text as png image

    Args:
                    report_str (str): text string to be saved
                    filename (str): image save location

    output:
                    None
    """
    _, ax = plt.subplots(figsize=(8, 3))  # Set the size of the image
    ax.axis('off')  # Remove axes

    # Add the classification report as text
    plt.text(
        0,
        1,
        report_str,
        ha='left',
        va='center',
        fontsize=10,
        family='monospace')

    # Save the plot as an image file
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def classification_report_image(y_train, y_test,
                                y_train_preds, y_test_preds,
                                img_name):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds: training predictions
                    y_test_preds: test predictions
                    img_name: image name
    output:
                     None
    '''

    report_str = img_name
    report_str += '\ntest results\n'
    report_str += classification_report(y_test, y_test_preds)
    report_str += '\ntrain results\n'
    report_str += classification_report(y_train, y_train_preds)
    save_text_as_image(report_str, "images/results/" + img_name)


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in output_pth
    input:
                    model: model object containing feature_importances_
                    x_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                     None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    save_plot("random_forests_feature_importance", output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
                      x_train: x training data
                      x_test: x testing data
                      y_train: y training data
                      y_test: y testing data
    output:
                      None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='newton-cholesky', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': [1, 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        error_score='raise')
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # classification report image for random forests
    classification_report_image(y_train, y_test,
                                y_train_preds_rf, y_test_preds_rf,
                                "random_forests_report.png")

    # classification report image for logistic regression
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_test_preds_lr,
                                "logistic_regression_report.png")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def roc_plot(lr_mod, rfc_mod, x_test, y_test, output_pth):
    """
    plots ROC curves for random forest classifier and logistic regression
    and saves into a file.

    input:
            lr_mod: model for logistic regression
            rfc_mod: model for random forest classifier
            x_test: x test data
            y_test: y test data
            output_pth: path to store the figure

    output:
            None
    """
    lrc_plot = RocCurveDisplay.from_estimator(lr_mod, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    _ = RocCurveDisplay.from_estimator(
        rfc_mod, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)

    save_plot("roc_curve_result", output_pth)


if __name__ == "__main__":

    # Import data
    dataframe = import_data("data/bank_data.csv")

    # Perform EDA and store plots
    perform_eda(dataframe)

    # Peform data encoding
    dataframe = encoder_helper(dataframe)

    # Perform training and test split
    Xtrain, Xtest, ytrain, ytest = perform_feature_engineering(dataframe)

    # Train the model
    train_models(Xtrain, Xtest, ytrain, ytest)

    # Load trained models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    # Plot features imporance and store the plots under 'images/results'
    feature_importance_plot(rfc_model, pd.concat(
        [Xtrain, Xtest]), "images/results")

    # Plot ROC and store plots
    roc_plot(lr_model, rfc_model, Xtest, ytest, "images/results")
