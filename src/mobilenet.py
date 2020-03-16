import torch
from torchviz import make_dot
import urllib

url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)


model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
model.eval()


from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
# I think this resized the picture and does some processing on pixels.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))

# We load the labels for the classes from a file
import json

class_idx = json.load(open("imagenet_class_index.json", 'r'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

values, indices = output[0].sort()
for idx in indices[-10:]:
    print(idx2label[idx])
