# K Nearest Neighbors Project

Welcome to the K Nearest Neighbors (KNN) Project! This project involves working with a dataset and applying the KNN algorithm for classification. Here's an overview of what's happening in this project:

## Import Libraries

We start by importing necessary Python libraries, including pandas, seaborn, numpy, and matplotlib, to work with data and visualization.

## Get the Data

We read the dataset from a CSV file called 'KNN_Project_Data' into a pandas DataFrame. The first few rows of the DataFrame are displayed to give an initial glimpse of the data.

# Heart Disease Prediction

This repository contains data analysis and machine learning models for predicting heart disease based on a dataset with patient health attributes. The project uses classification models, including K-Nearest Neighbors (KNN), to predict the presence or absence of heart disease.

## Data Overview

The dataset consists of 303 patient records with the following attributes:
- `age`: Age of the patient
- `sex`: Gender (0 for female, 1 for male)
- `cp`: Chest pain type
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol level
- `fbs`: Fasting blood sugar (1 if > 120 mg/dl, 0 otherwise)
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (1 for yes, 0 for no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels (0-3) colored by fluoroscopy
- `thal`: Thalassemia type
- `target`: Presence of heart disease (1 for yes, 0 for no)

## Data Insights

- The dataset contains 303 records with no missing values.
- Data types include integers and floats, suitable for machine learning.
- Descriptive statistics reveal central tendency and variability.
- A correlation matrix highlights the relationships between features, with `cp` showing a strong positive correlation with heart disease.
- Visualizations, such as a heatmap and histograms, provide insights into data distributions.

## Model Development and Results

### K-Nearest Neighbors (KNN) Model

- The KNN model was trained and evaluated using cross-validation.
- The optimal number of neighbors (K) was determined.
- Cross-validation accuracy: ~0.6933.
- Test set accuracy: ~0.623.
- Sample test cases demonstrate the model's ability to predict heart disease likelihood based on input features.




## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed using seaborn to create a pairplot. The 'TARGET CLASS' column is used as the hue, allowing us to visualize relationships between features and the target class.

## Standardize the Variables

To prepare the data for the KNN algorithm, the features are standardized using Scikit-learn's StandardScaler. This ensures that all features have the same scale, which is essential for KNN.

## Train-Test Split

The data is split into a training set and a testing set using the train_test_split function from Scikit-learn. This division is critical for evaluating the model's performance.

## Using K Nearest Neighbors (KNN)

We employ the K Nearest Neighbors (KNN) algorithm for classification. A KNN model is created with a specified number of neighbors (in this case, K=1). The model is then fitted to the training data.

## Predictions and Evaluations

The KNN model is evaluated by making predictions on the test data. A confusion matrix and a classification report are generated to assess the model's performance.

## Choosing a K Value

The optimal K value for the KNN model is determined using the elbow method. A for loop trains KNN models with different K values and tracks the error rate for each model.

## Retrain with New K Value

The KNN model is retrained with the selected K value. A new confusion matrix and classification report are created to evaluate the model's performance with the chosen K value.

This project is a practical exercise in data preprocessing, model training, and evaluation using the KNN algorithm. The code and explanations provided guide you through each step of the project.

## Conclusion

This project offers insights into heart disease prediction using data analysis and a K-Nearest Neighbors classification model. The model shows reasonable accuracy, but further optimization and evaluation are possible. Additional metrics, feature engineering, and model selection can enhance predictive performance.

For more details and code, please refer to the Jupyter Notebook or Python script in this repository.
