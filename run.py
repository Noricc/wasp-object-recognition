import torch.nn
from torch.utils.data import random_split  # For splitting into sets.
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets
import os
import cv2
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import torchvision as tv
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


from torchvision import transforms
from cv2.cv2 import putText

# No idea what these are.
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

classes_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
                 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad',
                 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry',
                 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
                 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi',
                 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza',
                 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream',
                 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons',
                 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella',
                 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
                 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
                 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara',
                 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
                 'tuna_tartare', 'waffles']

model_food = torch.hub.load('pytorch/vision:v0.5.0',
                            'mobilenet_v2', pretrained=True)


for param in model_food.parameters():
    param.requires_grad = False

num_features = model_food.classifier[1].in_features
out_features = 101

model_food.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(in_features=num_features,
                    out_features=out_features,
                    bias=True),
    torch.nn.ReLU(inplace=True))
PATH = "./food_model_2e7"
model_food.load_state_dict(torch.load(PATH))
model_food.eval()

cap = cv2.VideoCapture(0)

currentFrame = 0
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    input_image = pil_im
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    with torch.no_grad():
        outputs = model_food(input_batch)
        c, predicted = torch.max(outputs, 1)
        if(torch.max(outputs)) < 4:
            text = text = "Don't know"
        else:
            text = 'Predicted: '+' '.join('%5s' % classes_names[predicted[j]]
                                          for j in range(1))

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        textX = (frame.shape[1] - textsize[0]) / 2
        textY = textsize[1]-2
        cv2.putText(frame, text, (int(textX), int(textY)),
                    font, 1, (0, 0, 128), 2)

        cv2.imshow('WASP', frame)
        currentFrame += 1

cap.release()
cv2.destroyAllWindows()
