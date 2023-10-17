# K Nearest Neighbors Project

Welcome to the K Nearest Neighbors (KNN) Project! This project involves working with a dataset and applying the KNN algorithm for classification. Here's an overview of what's happening in this project:

## Import Libraries

We start by importing necessary Python libraries, including pandas, seaborn, numpy, and matplotlib, to work with data and visualization.

## Get the Data

We read the dataset from a CSV file called 'KNN_Project_Data' into a pandas DataFrame. The first few rows of the DataFrame are displayed to give an initial glimpse of the data.

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
