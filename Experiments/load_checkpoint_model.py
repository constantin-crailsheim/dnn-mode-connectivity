# %%

import torch
import sys
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
  
# Load files from parent directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import curves
import models
import data
from models.basiccnn import BasicCNN

# %%

model = "BasicCNN"
dataset = "MNIST"

loaders, num_classes = data.loaders(
    dataset = dataset,
    path = '/home/ubuntu/Project/Repos/dnn-mode-connectivity/tmp/data',
    batch_size = 1,
    num_workers = 1,
    transform_name = model,
    use_test = "True"
)

architecture = getattr(models, model)

base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
checkpoint = torch.load("/home/ubuntu/Project/Repos/dnn-mode-connectivity/tmp/MNIST_BasicCNN/checkpoints_model_1/checkpoint-4.pt")
base_model.load_state_dict(checkpoint['model_state'])

base_model.eval()


# %%

model = "ConvFC"
dataset = "CIFAR10"

loaders, num_classes = data.loaders(
    dataset = dataset,
    path = '/home/ubuntu/Project/Repos/dnn-mode-connectivity/tmp/data',
    batch_size = 1,
    num_workers = 1,
    transform_name = model,
    use_test = "False"
)

architecture = getattr(models, model)

base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
checkpoint = torch.load("/home/ubuntu/Project/Repos/dnn-mode-connectivity/tmp/CIFAR10_ConvFC/checkpoints_model_1/checkpoint-4.pt")
base_model.load_state_dict(checkpoint['model_state'])

base_model.eval()


# %%
f = gzip.open('/home/ubuntu/Project/Repos/dnn-mode-connectivity/tmp/data/mnist/MNIST/raw/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 5

f.read(16)
buf = f.read(image_size * image_size * num_images)
images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
images = images.reshape(num_images, image_size, image_size, 1)
image = np.asarray(data[0]).squeeze()
plt.imshow(image)
plt.show()


# %% 

iter = 0
iter_max = 10
for input, target in loaders['test']:
    iter += 1
    device = torch.device('cpu')
    input = input.to(device)
    image = np.asarray(input).squeeze()
    plt.imshow(image)
    plt.show()
    output = base_model(input)
    pred = output.data.argmax(1, keepdim=True)
    print(pred)
    if iter > iter_max:
        break


# %%
