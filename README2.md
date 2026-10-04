# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn.

## Project Overview

Customer churn is an important business problem for telecom companies. The goal of this project is to analyze customer data and build classification models that can identify customers who are likely to leave the company.

## Dataset

The dataset contains customer information such as:

* Customer demographics
* Tenure
* Contract type
* Internet services
* Monthly charges
* Total charges
* Churn status

## Machine Learning Workflow

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Categorical Encoding
* Train/Test Split
* Feature Scaling
* Feature Selection
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Cross-Validation
* Class Imbalance Handling
* Confusion Matrix
* ROC-AUC Evaluation
* Model Comparison

## Models

Three main classification algorithms were evaluated:

* Logistic Regression
* KNN
* SVM

A balanced version of Logistic Regression was also tested using `class_weight="balanced"` to improve detection of churned customers.

## Results

| Model                        | Churn Recall |
| ---------------------------- | -----------: |
| Balanced Logistic Regression |    **80.1%** |
| Logistic Regression          |        54.5% |
| KNN                          |        50.6% |
| SVM                          |        50.2% |

The **Balanced Logistic Regression** model achieved the best churn recall.

### Final Model Performance

* Churn Recall (5-Fold CV): **80.1%**
* ROC-AUC: **0.841**
* Test Accuracy: **74%**

## Conclusion

The balanced Logistic Regression model was selected as the final model because detecting customers who are likely to churn was the main objective of the project.

Feature selection was also tested with KNN. Reducing the feature space from 45 to 10 features slightly improved accuracy but reduced churn recall, showing that feature selection does not always improve model performance.

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
