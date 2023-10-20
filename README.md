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

## Predicting Heart Disease

In this project, the KNN model is utilized to predict the presence or absence of heart disease in patients. Here's how the prediction process works:

1. **Training the KNN Model**: The KNN algorithm is trained on a subset of the dataset known as the training set. The model learns the relationships between the patient attributes and the presence of heart disease based on this training data.

2. **Making Predictions**: Once the model is trained, it can make predictions on new, unseen data. In this case, the model can predict whether a patient is likely to have heart disease or not based on their health attributes.

3. **Model Evaluation**: The model's accuracy is evaluated by comparing its predictions to actual outcomes on a separate test dataset. The accuracy indicates how well the model can generalize and make accurate predictions on new data.

4. **Optimal K Value Selection**: To improve the model's accuracy, the optimal number of neighbors (K) is determined using the elbow method. This process helps select the most suitable K value for the KNN algorithm.

## Using K Nearest Neighbors (KNN)

We employ the K Nearest Neighbors (KNN) algorithm for classification. In this project, the specific features 'cp' (Chest Pain Type), 'thalach' (Maximum Heart Rate Achieved), and 'slope' (Slope of the Peak Exercise ST Segment) are used to predict the likelihood of heart disease presence based on these attributes. Here's how these features are utilized in the KNN model:

- **Chest Pain Type (cp)**: The 'cp' feature represents the type of chest pain experienced by the patient. This variable contains categorical values (0, 1, 2, 3) indicating different types of chest pain. The KNN model uses this information to find patterns in the relationship between chest pain type and heart disease.

- **Maximum Heart Rate Achieved (thalach)**: 'thalach' signifies the maximum heart rate achieved during a patient's exercise stress test. This feature provides insights into the patient's cardiovascular response to exercise. The KNN model considers the maximum heart rate achieved as a key factor in heart disease prediction.

- **Slope of the Peak Exercise ST Segment (slope)**: The 'slope' feature describes the slope of the peak exercise ST segment on an electrocardiogram. This variable reflects how the ST segment changes during peak exercise. The KNN algorithm takes into account this information to assess its impact on the likelihood of heart disease.

The KNN model analyzes the relationships between these features and the target variable ('target': 1 for presence of heart disease, 0 for absence) by considering the values of these features in the dataset. The classification process is based on the similarity of the features of the unknown data points (e.g., 'test1' and 'test2') to the features of the known data points in the training set. Once the model has been trained, it can make predictions about the presence or absence of heart disease for new cases based on these three features.

This project provides a practical example of using KNN to predict heart disease and demonstrates how specific features like 'cp,' 'thalach,' and 'slope' can play a crucial role in making these predictions.


## How KNN Helps

The K-Nearest Neighbors (KNN) algorithm is a powerful tool for classification tasks, such as heart disease prediction in this project. Here's how KNN contributes to this use case:

- **Data-Driven Classification**: KNN makes predictions based on the similarity of new data points to existing data. In this project, it identifies patients with similar health attributes and uses their outcomes to predict whether a new patient is likely to have heart disease.

- **Flexibility**: KNN is a versatile algorithm that can be applied to a wide range of classification tasks. It doesn't assume any specific form of the underlying data distribution, making it suitable for various use cases.

- **Interpretability**: KNN models are relatively easy to interpret. They provide insights into why a particular prediction was made by showing the nearest neighbors and their outcomes.

- **Model Tuning**: The choice of the number of neighbors (K) allows for model tuning. By experimenting with different K values, the model's performance can be optimized for specific use cases.

KNN's ability to consider the local neighborhood of data points and make predictions based on their similarity makes it a valuable tool in healthcare applications like heart disease prediction.

This project is a practical exercise in data preprocessing, model training, and evaluation using the KNN algorithm. The code and explanations provided guide you through each step of the project.

## Conclusion

This project offers insights into heart disease prediction using data analysis and a K-Nearest Neighbors classification model. The model shows reasonable accuracy, but further optimization and evaluation are possible. Additional metrics, feature engineering, and model selection can enhance predictive performance.

For more details and code, please refer to the Jupyter Notebook or Python script in this repository.
