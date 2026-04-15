# Stroke Prediction Project

## Project Overview
This project is a supervised machine learning study for stroke prediction based on patient health and demographic data. The notebook `main1.ipynb` performs data loading, exploratory data analysis, preprocessing, model training, and model comparison using common classification algorithms.

## Files Included
- `main1.ipynb` - The main Python notebook containing the full workflow, code cells, and analysis.
- `full_data.csv` - The dataset used in the notebook.
- `explain.txt` - Additional explanation notes or project-related information.
- `التحويلات.txt` - Transformation notes or supplementary preprocessing details.

## Dataset Description
The dataset includes patient information and a binary label indicating stroke occurrence. Key features used in the notebook include:
- `gender` - Patient gender, encoded as 1 for Male and 0 for Female.
- `age` - Patient age.
- `hypertension` - Whether the patient has hypertension.
- `heart_disease` - Whether the patient has heart disease.
- `ever_married` - Marital status, encoded as 1 for Yes and 0 for No.
- `work_type` - Type of work, encoded using `LabelEncoder`.
- `Residence_type` - Residence type, encoded as 1 for Urban and 0 for Rural.
- `avg_glucose_level` - Average blood glucose level.
- `bmi` - Body mass index.
- `smoking_status` - Smoking status, encoded using `LabelEncoder`.
- `stroke` - Target variable with 0 for no stroke and 1 for stroke.

## Notebook Workflow
The notebook follows a standard data science workflow:
1. Import required libraries.
2. Load the dataset from `full_data.csv`.
3. Inspect the dataset structure, missing values, and initial statistics.
4. Encode categorical variables and transform text fields to numeric values.
5. Visualize class balance and feature distributions.
6. Detect and handle outliers for numeric columns.
7. Explore relationships between features and the target label.
8. Train and evaluate a Random Forest classifier.
9. Compare performance across multiple classification algorithms.

## Preprocessing Steps
- `gender` is mapped to numeric values: `Male` → 1, `Female` → 0.
- `ever_married` is mapped: `Yes` → 1, `No` → 0.
- `Residence_type` is mapped: `Urban` → 1, `Rural` → 0.
- `work_type` and `smoking_status` are encoded using `LabelEncoder`.
- A `RobustScaler` is applied to scale numeric features before model training.
- Extreme values in `avg_glucose_level` are clipped using the dataset median for values greater than 200.

## Exploratory Data Analysis
The notebook performs EDA with:
- Display of dataset head and summary statistics.
- Pie chart showing the stroke class distribution.
- Value counts and histograms for each feature.
- Boxplots for numeric features to identify outliers.
- Countplots showing feature relationships with the target `stroke` label.

## Modeling and Evaluation
### Random Forest Model
- A `RandomForestClassifier` is trained using balanced class weights.
- Performance metrics reported include:
  - Accuracy
  - Recall
  - ROC-AUC
  - Confusion matrix
  - Classification report
- The number of trees and individual tree depths are also examined.

### Additional Model Comparison
The notebook also evaluates several models using the same preprocessed dataset:
- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Each model is evaluated using accuracy, recall, and ROC-AUC on the same test split.

## How to Run
1. Open `main1.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure `full_data.csv` is in the same folder as the notebook.
3. Run all cells in order.
4. Review plots, evaluation metrics, and model comparison output.

## Requirements
The notebook uses the following Python packages:
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install dependencies using pip if needed:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Notes
- The notebook currently assumes the dataset contains no missing values that require explicit imputation beyond the applied transformations.
- The selected features are numeric and categorical columns that are directly relevant to stroke prediction.
- Model performance can be improved further by exploring feature engineering, missing value handling, and cross-validation.

## Summary
This notebook provides a complete pipeline for building a stroke prediction model from raw data to evaluation. It includes data preprocessing, visualization, outlier handling, and comparison of multiple classification models to determine the best performing approach.
