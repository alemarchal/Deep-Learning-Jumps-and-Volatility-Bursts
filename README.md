# Deep-Learning-Jumps-and-Volatility-Bursts

## Introduction

This repo implements the methodology of the paper **[Deep Learning, Jumps, and Volatility Bursts](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3452933)**.

The code is divided into 3 main parts.

(1) Training an LSTM network using simulated data that mimic stylised facts of equity price time-series.
The network is trained to solve a classification problem with two classes: {jump, no jump} for each individual return.
This is implemented in the first part of the file *NN_jumps_classification*.


(2) Evaluate the performance of the network on out-of-sample simulated data.
This is implemented in the second part of *NN_jumps_classification*.


(3) Deploy the network on real market data.
The jump detection method is here used for disentangling continuous volatility from jump volatility and improve realised variance predictions.


If you use another frequency than 2-min, you will either have to rescale your returns or train a new network.


## Important files
*NN_jumps_classification*: generates training data (via Monte Carlo simulations), trains an LSTM network, evaluates this network on simulated data, and classifies real data.


*NN_jumps_applications*: it uses a trained network in order to implement the volatility forecasting exercise presented in our paper.

The remaining files are helper functions.


If you simply want to use the neural networks presented in the paper you can simply load 
the file *TrainedNeuralNetwork.mat* for instance and use *classify(net,XTest)* where *XTest* contains your test data.

## Hyperparameters

Two different architectures are provided.

(i) A bidirectional LSTM named *TrainedNeuralNetwork.mat*. This model uses past and future information when classifying a return.
(ii) A unidirectional LSTM named *TrainedNeuralNetwork_Unidirectional.mat*. With this model only the past information is fed to the network.

To examine the hyperparameters in details, please load the network and execute *analyzeNetwork(net)* in the Matlab console.
This will give all you information on the number of layers, units and activation functions that were used.

We tried with a few other architectures at random (for instance add an LSTM layer or increase the number of units) but the performance was similar. In general, a rather shallow network already works well.

Note that if you wish you train your own network, the classes are **highly** imbalanced.
