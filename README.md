# Assessing_CNN_RF_and_SVM_Geospatial_AI_Deep_Learning_models

![model_performance_pyplot](https://github.com/Kwakuopokuware401/Assessing_CNN_RF_and_SVM_Geospatial_AI_Deep_Learning_models/assets/94206249/8fd53e2d-4ab9-4711-bb6a-54b7b4684f6c)

Author: Kwaku Opoku-Ware (kwakuopokuware401@gmail.com)

# Description
Explore the power of Geospatial AI with an in-depth assessment of CNN, RF, and SVM Deep Learning models. This GitHub repository contains code and resources for comparing and contrasting these models in geospatial applications. Enhance your understanding of AI-driven geospatial analysis.

# Image Classification with CNN, RF, and SVM

This project compares different machine learning models - Convolutional Neural Network (CNN), Random Forest (RF), and Support Vector Machine (SVM) - for multi-class image classification. The models are evaluated and compared using various metrics like accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

## Models

The following models are implemented and evaluated:

## - **Convolutional Neural Network (CNN)** - A deep neural network with convolutional and pooling layers

This code is an example of using a convolutional neural network (CNN) for image classification using the TensorFlow and Keras libraries. Here's a simplified explanation of each part of the code:

1.	Importing libraries:

•	numpy (imported as np) is a library for numerical computations.

•	tensorflow (imported as tf) is a popular machine learning library.

•	keras is a high-level neural networks API that runs on top of TensorFlow.

2.	Defining data parameters:

•	image_height: Specifies the desired height of the input images.

•	image_width: Specifies the desired width of the input images.

•	num_channels: Specifies the number of color channels in the input images (e.g., 3 for RGB).

•	num_classes: Specifies the number of categories/classes for image classification.

3.	Defining the model architecture:

•	The model is defined as a sequential stack of layers using the keras.Sequential class.

•	Conv2D layers perform convolutional operations to extract features from the images.

•	MaxPooling2D layers downsample the input to reduce spatial dimensions.

•	Flatten layer flattens the 2D output into a 1D vector.

•	Dense layers are fully connected layers that perform classification.

•	Activation functions like relu and softmax introduce non-linearity to the model.

4.	Compiling the model:

•	The model is compiled with an optimizer, adam, which is an algorithm for training the model.

•	SparseCategoricalCrossentropy is used as the loss function, which measures the difference between predicted and true labels.

•	accuracy is used as a metric to evaluate the model during training.

5.	Generating training data:

•	Random placeholder training images and labels are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual training data.

6.	Training the model:

•	The fit method is used to train the model on the training images and labels.

•	The model is trained for a specified number of epochs (iterations) with a batch size of 32.

7.	Generating testing data:

•	Random placeholder testing images and labels are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual testing data.

8.	Evaluating the model:

•	The evaluate method is used to evaluate the trained model on the testing images and labels.

•	The test loss and accuracy are computed and stored in test_loss and test_acc, respectively.

•	The test accuracy is then printed to the console.

9.	Generating new images for classification:

•	Random placeholder new images are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual new images.

10.	Classifying the new images:

•	The predict method is used to classify the new images using the trained model.

•	The predicted class probabilities are stored in the predictions variable.

Note: This code uses random data for demonstration purposes. To use your own data, replace the random data generation with your actual image data and labels.
I hope this explanation helps! Let me know if you have any further questions.

  
## - **Random Forest (RF)** - An ensemble of decision trees

In this script, we've replaced the CNN model with a Random Forest classifier from the sklearn (scikit-learn) library. Here's a summary of the modifications:

1.	Importing libraries:

•	numpy (imported as np) is a library for numerical computations.

•	RandomForestClassifier from sklearn.ensemble is used for the Random Forest model.

2.	Generating training data:

•	Random placeholder training images and labels are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual training data.

3.	Creating the Random Forest classifier:

•	The RandomForestClassifier is instantiated with n_estimators=100, which specifies the number of decision trees in the ensemble.

4.	Training the classifier:

•	The fit method is used to train the classifier on the training images and labels.

5.	Generating testing data:

•	Random placeholder testing images and labels are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual testing data.

6.	Evaluating the classifier:

•	The score method is used to evaluate the trained classifier on the testing images and labels.

•	The test accuracy is computed and stored in test_accuracy.

•	The test accuracy is then printed to the console.

7.	Generating new images for classification:

•	Random placeholder new images are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual new images.

8.	Classifying the new images:

•	The predict method is used to classify the new images using the trained Random Forest classifier.

•	The predicted labels are stored in the predictions variable.

Note: This code uses random data for demonstration purposes. Replace the random data generation with your actual image data and labels.
Feel free to customize the script based on your specific needs and actual data. Let me know if you have any further questions!


## - **Support Vector Machine (SVM)** - A linear SVM classifier 

In this script, we've used the SVM model from the sklearn.svm module. Here's a breakdown of the modifications:

1.	Importing libraries:

•	numpy (imported as np) is a library for numerical computations.

•	SVC from sklearn.svm is used for the SVM model.

2.	Generating training data:

•	Random placeholder training images and labels are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual training data.

3.	Creating the SVM classifier:

•	The SVC class is instantiated with kernel='linear', which specifies a linear kernel for the SVM.

4.	Training the classifier:

•	The fit method is used to train the classifier on the training images and labels.

5.	Generating testing data:

•	Random placeholder testing images and labels are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual testing data.

6.	Evaluating the classifier:

•	The score method is used to evaluate the trained classifier on the testing images and labels.

•	The test accuracy is computed and stored in test_accuracy.

•	The test accuracy is then printed to the console.

7.	Generating new images for classification:

•	Random placeholder new images are generated using np.random.rand for demonstration purposes.

•	Replace these placeholders with your actual new images.

8.	Classifying the new images:

•	The predict method is used to classify the new images using the trained SVM classifier.

•	The predicted labels are stored in the predictions variable.

Note: This code uses random data for demonstration purposes. Replace the random data generation with your actual image data and labels.
Feel free to customize the script according to your specific needs and actual data. Let me know if you have any further questions!


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

## Visualizations

The model evaluation metrics are also visualized as plots and charts using Matplotlib, Seaborn, Plotly, and Altair.
The complete source code and visualizations can be found in the accompanying files.
