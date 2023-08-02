import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse

from utils import visualize_test_predictions
from model import CNNModel, model_resnet18, model_efficientnet

import warnings
warnings.filterwarnings("ignore")

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default= 'Resnet',
    help='name of model to test: Resnet or Efficient')
args = parser.parse_args()

PATH_DATA = 'dataset/add_images'
                
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# list containing all the class labels
labels = ['field', 'road']

# initialize the model
name = args.model

if name == 'Resnet':
    model = model_resnet18() 
    link = 'output/model_resnet18.pth'
if name == 'Efficient':
    model = model_efficientnet()
    link = 'output/model_efficientnet.pth'

model = model.to(device)
checkpoint = torch.load(link, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# parameters for test images normalizing
test_mean =[0.5374407, 0.5403282, 0.438259]
test_std =[0.24362235, 0.2223072, 0.2965594]

# define preprocess transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=test_mean,std=test_std)
    ]
)

# import data
test_data = ImageFolder(
    root= PATH_DATA,
    transform= transform
)

#data loaders
test_loader = DataLoader(test_data,shuffle=True)

# predict and print pictures
visualize_test_predictions(model, test_loader,labels, num_images=10)
