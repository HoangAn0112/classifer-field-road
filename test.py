import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from utils import visualize_test_predictions
from model import CNNModel


PATH_DATA = 'dataset/test_images'
                
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# list containing all the class labels
labels = ['field', 'road']
# initialize the model and load the trained weights
model = CNNModel().to(device)
checkpoint = torch.load('output/model_baseline.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# parameters for normalizing
test_mean = (0.5374407, 0.5403282, 0.438259)
test_std = (0.24362235, 0.2223072, 0.2965594)

# define preprocess transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),  
    transforms.ToTensor(),
    transforms.Normalize(test_mean,test_std)]
)

# import data
test_data = ImageFolder(
    root= PATH_DATA,
    transform= transform
)

# training data loaders
test_loader = DataLoader(test_data)

visualize_test_predictions(model, test_loader,labels)