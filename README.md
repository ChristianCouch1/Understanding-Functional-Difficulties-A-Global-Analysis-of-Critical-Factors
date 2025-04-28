<details>
<summary><strong> 📌 Table of Contents</strong></summary>

- [Overview](#overview)
- [Background](#background)
- [Goals](#goals)

## Part I: Clean Datasets and Merge Datasets
- [Transform Data in Power BI](#transform-data-in-power-bi)
- [Importing Packages and Loading Data Into Python](#importing-packages-and-loading-data-into-python)
- [Fixing Attendance Rate Dataset in Python](#fixing-attendance-rate-dataset-in-python)

## Part II: Machine Learning / Statistical Modeling for Attendance Rate Dataset
- [Logistic Regression (Attendance Rate)](#logistic-regression-attendance-rate)
- [ROC Curve for Logistic Regression (Attendance Rate)](#roc-curve-for-logistic-regression-attendance-rate)
- [Cross Validation for Logistic Regression (Attendance Rate)](#cross-validation-for-logistic-regression-attendance-rate)
- [Random Forest Classification (Attendance Rate)](#random-forest-classification-attendance-rate)

## Part III: Combining Multiple Datasets and Cleaning Them
- [Merging the Datasets in Power BI](#merging-the-datasets-in-power-bi)
- [Importing New Updated Data Into Python](#importing-new-updated-data-into-python)
- [Cleaning the Merged Dataset In Python](#cleaning-the-merged-dataset-in-python)

## Part IV: Machine Learning / Statistical Modeling for Merged Dataset
- [Logistic Regression (Merged Dataset)](#logistic-regression-merged-dataset)
- [ROC Curve for Logistic Regression (Merged Dataset)](#roc-curve-for-logistic-regression-merged-dataset)
- [Cross Validation for Logistic Regression (Merged Dataset)](#cross-validation-for-logistic-regression-merged-dataset)
- [Random Forest Classification (Merged Dataset)](#random-forest-classification-merged-dataset)
- [ROC Curve for Random Forest (Merged Dataset)](#roc-curve-for-random-forest-merged-dataset)

## Part V: Visualization Tools
- [Power BI](#power-bi)

</details>


# Background
This project was completed for my Senior Project/Portfolio class. It focuses on understanding the key factors that predict whether individuals from different countries experience functional difficulties. Using data from UNICEF's Global Database on Education for Children with Disabilities, the study explores how education and demographic factors may relate to functional challenges.

# Overview
This research investigates the key predictors of functional difficulties among individuals across different countries. Specifically, it analyzes the impact of Adjusted Net Attendance Rate (ANSR), foundational reading test scores, foundational numeracy scores, and sex. Logistic regression and a Random Forest classifier are applied to the UNICEF dataset to distinguish between individuals with and without functional difficulties and to evaluate the extent of any skill gaps.

# Goals
- Determine which educational and demographic factors (such as ANSR, foundational reading and numeracy skills, and sex) are most strongly associated with functional difficulties
- Build and compare logistic regression and Random Forest models to classify individuals with and without functional difficulties
- Develop a Random Forest model to further improve classification accuracy and feature importance analysis
- Evaluate model performance using ROC curves and cross-validation techniques

## Part I: Clean Datasets, and Merge Datasets
### Transform Data in Power BI
- The Data was Obtain from UNICEF website
- It was an Excel file with multiple sheets, including data on attendance rates, foundational reading scores, foundational numeracy scores, and out-of-school rates.
- Put the different Files into Power BI and transform the data removing rows and changing the table headings to make them uniform.
### Importing Packages and Loading Data Into Python
Below are the packages used for this project:
```
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
from sklearn.discriminant_analysis import \
     (LinearDiscriminantAnalysis as LDA,
      QuadraticDiscriminantAnalysis as QDA)
from ISLP import confusion_table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from sklearn.metrics import roc_auc_score
from ISLP.models import sklearn_sm
from sklearn.ensemble import RandomForestClassifier
```

### Fixing Attendance Rate Dataset in Python
-Reformatted the dataset by restructuring some columns into 'With' and 'Without' rows, indicating whether an entry has functional differences or not.
```
def create_anar_summary_dataframe(df):

    # Filter for "Without"
    df_without = df.dropna(subset=['ANAR Point Estimate Without'])
    df_without = df_without.rename(columns={
        'ANAR Point Estimate Without': 'Point Estimate',
        'ANAR Upper Limit Without': 'Upper Limit',
        'ANAR Lower Limit Without': 'Lower Limit'
    })
    df_without['Functional Difficulties'] = 'Without'

    # Filter for "With"
    df_with = df.dropna(subset=['ANAR Point Estimate With Functional Difficulties '])
    df_with = df_with.rename(columns={
        'ANAR Point Estimate With Functional Difficulties ': 'Point Estimate',
        'ANAR Upper Limit With Functional Difficulties': 'Upper Limit',
        'ANAR Lower Limit With Functional Difficulties Limit': 'Lower Limit'
    })
    df_with['Functional Difficulties'] = 'With'

    # combineing the Datasets
    df_combined = pd.concat([df_without, df_with], ignore_index=True)
```
- Finished removing null values and excluded countries like Belarus that do not accurately report their numbers, in order to reduce the likelihood of unreliable or inaccurate results.

## Part II: Machine Learning / Statistical Modeling for Attendance Rate Dataset

### Logistic Regression (Attendance Rate)
- Used one-hot encoding for gender categories ('Male', 'Female').
- Applied a Logistic Regression model to classify entries into "With" or "Without" functional difficulties.
- Evaluated model performance using accuracy, confusion matrix, and classification report.
- PUT PICTURE HERE FOR THE CONFUSTION MATRIX
- Accuracy: Insert Logistic Regression accuracy here
### ROC Curve for Logistic Regression (Attendance Rate)
- Plotted the ROC curve for the Logistic Regression model.
![ROC Curve Log REF ANAR](https://github.com/user-attachments/assets/7b74eb31-da64-416d-b321-66c5ff07d50c)
- ROC AUC Score: .88
### Cross Validation for Logistic Regression (Attendance Rate)
- Performed 5-fold cross-validation to check model stability and reduce overfitting.
![Cross Val for Log ](https://github.com/user-attachments/assets/2915512f-b5f6-434e-9eae-5b6b573fb273)
- Best Fold 2 CV Score: 0.925
### Random Forest Classification (Attendance Rate)
- Trained a Random Forest Classifier using 70% training and 30% testing split
- Accuracy: Insert Random Forest accuracy here
## Part III: Combining Multiple Datasets and Cleaning Them
### Merging the Datasets in Power BI
- Merging the data into one datset to have it all together and combine them by having the country and catgory as a comine key. 
- Merging the datasets into one by combining them based on a composite key of "Country" and "Category".
- Insert PIC of Power BI
### Importing New Updated Data Into Python
- Exporting the appended data from power BI
- Imported all relevant data into Python, creating a single combined dataset that includes foundational reading scores, numeracy scores, attendance rates, and gender (sex) categories ### Cleaning the Merged Dataset In Python
- Performed the same process as with the previous data, formatting the dataset by restructuring certain columns into 'With' and 'Without' rows to indicate whether an entry has functional difficulties or not.
## Part IV: Machine Learning / Statistical Modeling for Merged Dataset
### Logistic Regression (Merged Dataset)
- Applied a Logistic Regression model to classify entries into "With" or "Without" functional difficulties.
- Evaluated model performance using accuracy, confusion matrix, and classification report.
- PUT PICTURE HERE FOR THE CONFUSTION MATRIX
- Accuracy: Insert Logistic Regression accuracy here
### ROC Curve for Logistic Regression (Merged Dataset)
- Plotted the ROC curve for the Logistic Regression model.
- PUT PICTURE HERE FOR THE ROC CURVE
![ROC Curve for Logistic Regression for Merge](https://github.com/user-attachments/assets/f6845aec-b395-45b8-af29-9944e1a659d9)
- ROC AUC Score: .91
### Cross Validation for Logistic Regression (Merged Dataset)
- Performed 5-fold cross-validation to check model stability and reduce overfitting.
![Cross Validation for Logestic Regresson for Merge](https://github.com/user-attachments/assets/55a9292d-7a2d-436e-924e-01f659e5e89a)
Best Fold 2 CV Score: 0.989
### Random Forest Classification (Merged Dataset)
- Trained a Random Forest Classifier using 70% training and 30% testing split
- Accuracy: Insert Random Forest accuracy here
![Feature Impordance for Random Forest](https://github.com/user-attachments/assets/004f379c-bba6-4dfe-bc91-7550d0579496)
### ROC Curve for Random Forest (Merged Dataset)
- Plotted the ROC curve for the random forest model.
![ROC Curve for Random Forest for Merge](https://github.com/user-attachments/assets/9c72eee0-d393-472d-a712-7c6263ba599f)
- ROC AUC Score: .89
## Part IV: Visualization Tools
### Power BI
- Insert PIC of DASHBORED HERE



