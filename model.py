import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

# Define a simple CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.fc1 = nn.Linear(256, 2)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.fc1(x)
        return x

# tranfer learning resnet18
def model_resnet18():
    model = models.resnet18(pretrained=True)   #load resnet18 model
    num_features = model.fc.in_features     #extract fc layers features
    model.fc = nn.Linear(num_features, 2) #(num_of_class == 2)
    return model

# tranfer learning efficientnet_b0
def model_efficientnet():
    model = models.efficientnet_b0(pretrained=True)   #load efficientnet_b0 model
    model.classifier[1] = nn.Linear(1280, 2) #(num_of_class == 2, number of input feature find in model print
    return model







