# Convolutional-Neural-Network---CIFAR-10-Image-Classification
CIFAR-10 is a dataset that consists of several images divided into the following 10 classes:

1. Airplanes
2. Cars
3. Birds
4. Cats
5. Deer
6. Dogs
7. Frogs
8. Horses
9. Ships
10. Trucks
The dataset stands for the Canadian Institute For Advanced Research (CIFAR)
CIFAR-10 is widely used for machine learning and computer vision applications.
The dataset consists of 60,000 32x32 color images and 6,000 images of each class.
Images have low resolution (32x32).
Data Source: https://www.cs.toronto.edu/~kriz/cifar.html

There are many new concepts I have learned during this project and it was fun learning neural networks and different python libraries. I will note down the important points of the concepts used in this project.
# Convolutional Neural Network:
Convolutional Neural Network is a deep learning algorithm which is used for recognizing images. This algorithm clusters images by similarity and perform object recognition within scenes. CNN uses unique feature of images to identify object that is placed on the image. 

# Keras:
Keras is a high-level, deep learning API developed by Google for implementing neural networks. It is written in Python and is used to make the implementation of neural networks easy. It also supports multiple backend neural network computation.
Keras allows you to switch between different back ends. The frameworks supported by Keras are,Tensorflow, Theano, PlaidML, MXNet, CNTK (Microsoft Cognitive Toolkit). It is used commercially by many companies like Netflix, Uber, Square, Yelp, etc which have deployed products in the public domain which are built using Keras. 
The research community for Keras is vast and highly developed. The documentation and help available are far more extensive than other deep learning frameworks. 

# Sequential Model - Keras:
The core idea of Sequential API is simply arranging the Keras layers in a sequential order and so, it is called Sequential API. As its name suggests it is one of the models that is used to investigate varied types of neural networks where the model gets in one input as feedback and expects an output as desired. The Keras API and library is incorporated with a sequential model to judge the entire simple model not the complex kind of model. It passes on the data and flows in sequential order from top to bottom approach till the data reaches at end of the model.
# Convolutional Layer(operation):
This process is main process for CNN. In this operation there is a feature detector or filter. This filter detects edges or specific shapes. Filter is placed top left of image and multiplied with value on same indices. After that all results are summed and this result is written to output matrix. Then filter slips to right to do this whole processes again and again. 
# Padding:
Convolutional operations when performed, decreases the size of the image and hence we need to apply padding to preserve the input size.
# Pooling:
This layer is used for reducing parameters and computating process. Also by using this layer features invariant to scale or orientation changes are detected and it prevents overfitting. 
# Flattening:
Flattening is taking matrix came from convolutional and pooling processes and turn it into one dimensional array. 
# Dropout:
Dropout is a regularization technique for reducing overfitting. It is called “dropout” because it drops out visible or hidden units in neural network.
# Adam Optimizer:
The Adam optimizer is a popular optimization algorithm used in machine learning for stochastic gradient descent (SGD)-based optimization. It stands for Adaptive Moment Estimation and combines the best parts of two other optimization algorithms, AdaGrad and RMSProp.
The key idea behind Adam is to use a combination of momentum and adaptive learning rates to converge to the minimum of the cost function more efficiently. During training, it uses the first and second moments of the gradients to change the learning rate on the fly.
# Tensorboard:
TensorBoard is a visualization tool provided with TensorFlow. This callback logs events for TensorBoard, including: Metrics summary plots, Training graph visualization, Weight histograms and Sampled profiling.


# Sources:
Machine Learning Practical workouts Course - Dr Ryan Ahmed, https://medium.com/@cdabakoglu/what-is-convolutional-neural-network-cnn-with-keras-cab447ad204c (Easy to understand and best for beginner to get a good grasp), https://www.educba.com/keras-sequential/, Geeksforgeeks, https://keras.io/api/callbacks/tensorboard/ etc.
