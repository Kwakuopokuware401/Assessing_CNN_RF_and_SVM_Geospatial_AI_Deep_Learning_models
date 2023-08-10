# Assessing_CNN_RF_and_SVM_Geospatial_AI_Deep_Learning_models

Author: Kwaku Opoku-Ware (kwakuopokuware401@gmail.com)

# Description
Explore the power of Geospatial AI with an in-depth assessment of CNN, RF, and SVM Deep Learning models. This GitHub repository contains code and resources for comparing and contrasting these models in geospatial applications. Enhance your understanding of AI-driven geospatial analysis.

# Image Classification with CNN, RF, and SVM

This project compares different machine learning models - Convolutional Neural Network (CNN), Random Forest (RF), and Support Vector Machine (SVM) - for multi-class image classification. The models are evaluated and compared using various metrics like accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

## Models

The following models are implemented and evaluated:

- **Convolutional Neural Network (CNN)** - A deep neural network with convolutional and pooling layers
- **Random Forest (RF)** - An ensemble of decision trees 
- **Support Vector Machine (SVM)** - A linear SVM classifier 

## Data

The models are trained and tested on randomly generated image data with the following properties:

- Image size: 128 x 128 pixels
- Number of channels: 3 (RGB)  
- Number of classes: 4
- Total images: 1000 (80% for training, 20% for testing)

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy** - Overall accuracy on test data
- **Precision** - Precision score (weighted average)
- **Recall** - Recall score (weighted average)
- **F1-score** - F1 score (weighted average)
- **Classification report** - Precision, recall, F1 score per class
- **Confusion matrix** - Confusion matrix on test data

## Results

The evaluation results on the test data are summarized below:

| Metric | CNN | Random Forest | SVM |
|-|-|-|-|
| Accuracy | 0.24 | 0.265 | 0.185 |
| Precision | 0.06 | 0.22 | 0.18 |
| Recall | 0.245 | 0.225 | 0.21 |
| F1-score | 0.096 | 0.208 | 0.177 |

## Visualizations

The model evaluation metrics are also visualized as plots and charts using Matplotlib, Seaborn, Plotly, and Altair.
The complete source code and visualizations can be found in the accompanying files.
