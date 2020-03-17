import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from torchviz import make_dot
import torchvision as tv
import urllib
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


from torchvision import transforms

# No idea what these are.
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

# Transforms for getting the same shape as in the original training data
preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])

import os
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.utils.data import random_split # For splitting into sets.

# Loading the images into a dataset.
data_dir = "../data/images"

# We only take the first 10000 pictures
indices = range(100)

dataset = datasets.ImageFolder(os.path.realpath(data_dir),
                              transform=preprocessing)

subset = dataset #Subset(dataset, indices)

size_train_data = int(0.8 * len(subset))
size_test_data = len(subset) - size_train_data

train_data, val_data = random_split(subset,
                                    (size_train_data, size_test_data))

# We make to dataloaders which only load a subset of the data
train_dataloader = DataLoader(train_data,
                              batch_size=16,
                              num_workers=20,
                              shuffle=True)
val_dataloader = DataLoader(val_data,
                            batch_size=16,
                            num_workers=20,
                            shuffle=True)

classes_names = dataset.classes

model_food = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)



# Creating the network to train.
import torch.nn

# Apparently PyTorch is quite "pythonic"
# I can just change the state of the network

# First, I tell PyTorch the parameters do not need to be trained.
# No need to compute gradient
for param in model_food.parameters():
    param.requires_grad = False

# Want to change the last layer, the "classifier" part.
# Number of features from lower level stays constant.
num_features = model_food.classifier[1].in_features
# Number of OUTPUT features, however, is changed
# We set it to the number of classes we have in food dataset
out_features = 101

model_food.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(in_features=num_features,
                    out_features=out_features,
                    bias=True),
    torch.nn.ReLU(inplace=True))


from torch.optim import SGD

# This is the algorithm for optimization
optimizer = SGD(model_food.classifier.parameters(),
               lr=0.001, momentum=0.9)

# This is the loss function (to measure how wrong the network is)
criterion = torch.nn.CrossEntropyLoss()

# Put the model in training mode
model_food.train() # Tchoo tchoo


n_epochs = 25
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients (why?)
        optimizer.zero_grad()

        # Forward phase
        outputs = model_food(inputs)
        loss = criterion(outputs, labels)
        # Backward phase (get the gradients)
        loss.backward()
        # Optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    torch.save(model_food.state_dict(), "food_model_2e{}".format(epoch))

print('Finished Training')
