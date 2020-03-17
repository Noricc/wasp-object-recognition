# This code is a mix-and-match from:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#

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

IMAGE_FOLDER="../data/images"

# Visualize some images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([mean_nums])
    # std = np.array([std_nums])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        plt.pause(0.001)  # Pause a bit so that plots are updated

    plt.show(block=True)



def train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print (phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        torch.save(model.state_dict(), "food_model_e{}".format(epoch))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main() :
    n_epochs = 25
    # set scale to 1 to train on the entire dataset
    scale = 0.001

    if len(sys.argv) == 3:
        n_epochs = int(sys.argv[1])
        scale = float(sys.argv[2])

    print("Training during {} epochs, using {}% of the dataset.".format(n_epochs, scale * 100))

    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # the folder where the images are
    img_folder = tv.datasets.ImageFolder(IMAGE_FOLDER, preprocess)

    n_img_train = int(len(img_folder) * 0.8)
    n_img_eval = len(img_folder) - n_img_train




    n_img_train = int(scale * n_img_train)
    n_img_eval = int(scale * n_img_eval)
    n_img_rest = len(img_folder) - n_img_train - n_img_eval


    [img_folder_train, img_folder_eval, _] = torch.utils.data.dataset.random_split(img_folder, [n_img_train, n_img_eval, n_img_rest])

    # load the training set in random order
    data_loader_train = torch.utils.data.DataLoader(img_folder_train, batch_size=16, num_workers=20,
                                              shuffle=True)

    data_loader_eval = torch.utils.data.DataLoader(img_folder_eval, batch_size=16, num_workers=20,
                                              shuffle=True)

    dataset_sizes = {'train' : n_img_train, 'val' : n_img_eval}
    dataloaders = {'train' : data_loader_train, 'val' : data_loader_eval}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the model from torch hub
    model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)

    # extract the class names
    class_names = img_folder.classes

    # model for training
    model_ft = model

    # num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train the model
    model_ft = train_model(device, dataloaders, dataset_sizes, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=n_epochs)

    torch.save(model_ft.state_dict(), "food_model")

    return


    # Grab some of the training data to visualize
    inputs, classes = next(iter(data_loader))

    print (classes)
    # Now we construct a grid from batch
    out = tv.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])



if __name__ == "__main__":
    main()
