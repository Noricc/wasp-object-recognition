---
title: "Presentation for WASP Autonomous Systems 1 assignment: Object Detection"
header-includes:
  - \hypersetup{colorlinks=false,
            allbordercolors={0 0 1},
            pdfborderstyle={/S/U/W 1}}
---

Application
===========

The application was to have a classifier that can classify different types of food.

Methods
=======

Training Data
-------------
[Food 101](https://www.kaggle.com/kmader/food41#1002850.jpg) data set from Kaggle.

Network specification
---------------------
The method was to use Transfer Learning on the MobileNet v2 network.
The machine learning framework which was used in PyTorch.

The MobileNet implementation is composed of two named components:

1. A stack of layers called **features**, which consists of 18 layers, mostly convolutional, which indicates they are used processing the image data.
2. One layer classed **classifier**,  which consists of two layers
    a. A dropout layer (which randomly drops values, setting them to zero during training).
    b. A linear transformation from the features to the classes.

We wanted to re-use the **features** part of the network, and only train the **classifier**.

The new **classifier** is the same as the original one, with a ReLU activation function appended:

a. A dropout layer (which randomly drops values, setting them to zero during training).
b. A linear transformation from the features to the classes.
c. A ReLU layer

Training
--------
We used Stochastic Gradient Descent with a momentum of 0.9 and a learning rate of 0.001

The loss function was Cross Entropy (see appendix).

Testing
-------

We measured the accuracy on the test set which has 20% of pictures in the full data set.

Results
=======

**WARNING** Update this after Alexandru made his runs

The network takes 1h 3minutes to be trained and yeilds 75% accuracy on the test set.

Repository
==========

- [Github](https://github.com/Noricc/wasp-object-recognition)

Participation in the group
==========================

* Noric: Jupyter notebook and documentation, based on a tutorial provided by Alexandru and PyTorch's documentation.
    * <https://stackabuse.com/image-classification-with-transfer-learning-and-pytorch/#settingupapretrainedmodel>
    * <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer>
* Idriss: Video setup and recording
* Hedieh: Presentation of work and results
* Alexandru: HPC Setup for training on the full training data.

We also want to thank Alexander Dürr for suggesting and explaining Transfer Learning.


Appendix
========

Cross Entropy
-------------

A loss function often used for classification in Machine Learning.
It is a way to compute how wrong a classifier is, when:

1. The classifier outputs a probability
2. The expected label is encoded as a one-hot vector, and therefore, the expected value for each class is either 0 or 1.

With M classes, and observation o, the cross-entropy is:
$$- \sum^{M}_{c = 1}y_{o, c} \log(p_{o,c})$$

Where $y_{o,c}$ is the binary indicator (0 or 1) if class label $c$ is the correct classification for observation $o$, and $p_{o,c}$ is the probability observation $o$ is of class $c$.

### Sources

- [The Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

Stochastic Gradient Descent
---------------------------

Doing gradient descent means you need to estimate the gradient with respect to the parameters of your model. Getting the gradient is very expensive, so you need to cut your dataset into smaller batches to get an estimate of the gradient.

If batches are too small, You get more error in your estimation of the gradient. If batches are too large, then you spend a lot of time trying to estimate your gradient.

Transfer Learning
-----------------

Transfer learning is based on the idea that you can re-use part of the convolutional network when it has been trained. Indeed, the lower layers of the network change very little between networks, as they detect edges, shapes, etc. This can always be useful. The top layer of the network, however, combine high level information that is more relevant to the particular data on which the network is trained.

The idea of transfer learning is to take an existing neural net, remove some top layers, and re-train on new data, using the bottom of the network as a pre-processing step.
