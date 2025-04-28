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
- [Logistic Regression](#logistic-regression)
- [ROC Curve for Logistic Regression](#roc-curve-for-logistic-regression)
- [Cross Validation for Logistic Regression](#cross-validation-for-logistic-regression)
- [Random Forest Classification](#random-forest-classification)
- [ROC Curve for Random Forest](#roc-curve-for-random-forest)

## Part III: Combining Multiple Datasets and Cleaning Them
- [Merging the Datasets in Power BI](#merging-the-datasets-in-power-bi)
- [Importing New Updated Data Into Python](#importing-new-updated-data-into-python)
- [Clean the Merged Dataset In Python](#clean-the-merged-dataset-in-python)

## Part IV: Machine Learning / Statistical Modeling for Merged Dataset
- [Logistic Regression](#logistic-regression-1)
- [ROC Curve for Logistic Regression](#roc-curve-for-logistic-regression-1)
- [Cross Validation for Logistic Regression](#cross-validation-for-logistic-regression-1)
- [Random Forest Classification](#random-forest-classification-1)
- [ROC Curve for Random Forest](#roc-curve-for-random-forest-1)

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

### Logistic Regression
-
-
### ROC Curve for Logistic Regression
-
-
### Cross Validation for Logistic Regression
-
-
### Random Forest Classifaction 
-
-
### ROC Curve for Random Forest
-
-
## Part III: Combining Multiple Datasets and Cleaning Them
### Merging the Datasets in Power BI
-
-

### Importing New updated Data Into Python
-
-
### Clean the Merged Dataset Into Python
-
-
## Part IV:  Machine Learning / Statistical Modeling for Merged Dataset
### Logistic Regression
-
-
### ROC Curve for Logistic Regression
-
-
### Cross Validation for Logistic Regression
-
-
### Random Forest Classifaction 
-
-
### ROC Curve for Random Forest
-
-
## Part IV: Visualization Tools
### Power BI
- -



