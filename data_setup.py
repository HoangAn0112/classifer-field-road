import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

PATH_DATA = 'dataset/train_images'

# set up parameters
batch_size = 8
validation_split = .2
shuffle_dataset = True
random_seed= 42

# parameters for pixel normalizing
train_mean = (0.45512244,0.46718585,0.41516596)
train_std= (0.24454536,0.23760799,0.27499264)

test_mean = (0.5374407, 0.5403282, 0.438259)
test_std = (0.24362235, 0.2223072, 0.2965594)

# import data
data = ImageFolder(
    root= PATH_DATA,
    transform=None
)

valid_size = int(validation_split*len(data))
train_size = len(data)-valid_size
train_dataset, valid_dataset = random_split(data, [train_size,valid_size])

# transform for train
train_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])
#     transforms.Normalize(train_mean, train_std)])

#transform for validation
valid_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])
#     transforms.Normalize(train_mean, train_std)])

# Apply the transformations to the datasets
train_dataset.dataset.transform = train_transform
valid_dataset.dataset.transform = valid_transform

# training data loaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True
)
# validation data loaders
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True
)