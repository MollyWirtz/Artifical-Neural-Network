# Artificial Neural Network
CS 4341 Project 2

The goal of this project was to build an artificial neural network for categorizing images. 
This program takes in images of hand-written numbers and outputs the predicted number the image represents. 

images.npy and labels.npy are files of 6,500 images and corresponding labels used to train, validate, and test the neural network. 

## Dependancies  
This project uses Keras, Tensorflow, MatplotLib, and SkLearn, and is written in Python. 

## About 
main.py contains code to build an ANN (artificial neural network) of 3+ fully connected layers: the input layer, the hidden layer(s), and the output layer. 
Stochastic gradient descent is used to train the ANN, and cross-validation has been implemented to split the dataset. 
Various experiments were run varying the epoch size, batch size, and hidden layer size. All these experiments, and their results on the accuracy, is documented in the included report. 

## Running
This program uses files "images.npy" and "labels.npy" located in the same directory as the executable, main.py. 
The output will show the ANN training over x amount of epochs, and will return the accuracy, error, confusion matrix, and accuracy plots of the training data set over the validation data set. 

