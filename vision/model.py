import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

class CentralisedCNN(nn.Module):
    # Default network
    def __init__(self, num_classes):
        super(CentralisedCNN, self).__init__()
        
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=self.num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        #out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        #out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


class CNN1(nn.Module):
    
    def __init__(self, num_classes):
        super(CNN1, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        #self.layer2 = nn.Sequential(
        #    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        #    nn.BatchNorm2d(64),
        #    nn.ReLU(),
        #    nn.MaxPool2d(2)
        #)
        
        self.fc1 = nn.Linear(in_features=128*3*14, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_classes)
        
    def forward(self, x):
        # also 4 clients, 
        out_temp = []
        x_split = torch.split(x,7,dim=2)
        for x_single in x_split:
            out0 = self.layer1(x_single)
            #out0 = self.layer2(out0)
            out_temp.append(out0)
        #out = self.layer1(x)
        #out = self.layer2(out)
        #out = out.view(out.size(0), -1)
        out = torch.cat(out_temp,dim=1)
        out = out.view(x_single.size()[0], -1)
        out = F.relu(self.fc1(out))
        #out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class MLP(nn.Module):
    # https://arxiv.org/pdf/2007.06081.pdf, following this paper
    # 4 clients
    # 7 layer of CNN and then fully connected layers
    # since fully connect layer, where to cut won't make any difference
    
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        
        # fc1 is the local module, so the infeatures for fc2 will be 64*4
        self.fc1 = nn.Linear(in_features=7*28, out_features=32)
        #self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=32*4, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=64)
        self.fc6 = nn.Linear(in_features=64, out_features=num_classes)
        
    def forward(self, x):
        # divided input into 4 clients
        out_temp = []
        x_split = torch.split(x,7,dim=2)
        for x_single in x_split:
            x_single = x_single.view(x_single.size()[0], -1)
            out_temp.append(F.relu(self.fc1(x_single)))
        
        # combine 4 clients
        out = torch.cat(out_temp,dim=1)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        return out

class CNN2(nn.Module):
    
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=1792, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_classes)
        
    def forward(self, x):
        # also 4 clients, 
        out_temp = []
        x_split = torch.split(x,7,dim=2)
        for x_single in x_split:
            out0 = self.layer1(x_single)
            out0 = self.layer2(out0)
            out_temp.append(out0)
        #out = self.layer1(x)
        #out = self.layer2(out)
        #out = out.view(out.size(0), -1)
        out = torch.cat(out_temp,dim=1)
        out = out.view(x_single.size()[0], -1)
        out = F.relu(self.fc1(out))
        #out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class CNN_and_CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_and_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        #self.fc0 = nn.Linear(in_features=128, out_features=64)
        self.fc1 = nn.Linear(in_features=448, out_features=256)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features= num_classes)
        
    def forward(self, x):
        # also 4 client
        # one CNN locally, NO linear layer, then CNN globally
        out_temp = []
        x_split = torch.split(x,7,dim=2)
        for x_single in x_split:
            out0 = self.layer1(x_single)
            out_temp.append(out0)
        out = torch.cat(out_temp,dim=1)
        # here need to add a linear layer

        out = self.layer2(out)
        out = out.view(x_single.size()[0], -1)
        out = F.relu(self.fc1(out))
        #out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class CNN_linear_CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_linear_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc0 = nn.Linear(in_features=128*3*14, out_features=128*3*14)#128x3x14
        self.fc1 = nn.Linear(in_features=448, out_features=256)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features= num_classes)
        
    def forward(self, x):
        # also 4 client
        # one CNN locally then linear layer then CNN globally
        out_temp = []
        x_split = torch.split(x,7,dim=2)
        for x_single in x_split:
            out0 = self.layer1(x_single)
            out_temp.append(out0)

        out = torch.cat(out_temp,dim=1)
        out = out.view(out.size(0), -1)

        # here need to add a linear layer
        out = self.fc0(out)
        out = torch.reshape(out, (out.size(0), 128, 3, 14))
        # then doing another CNN layer
        out = self.layer2(out)
        out = out.view(x_single.size()[0], -1)
        out = F.relu(self.fc1(out))
        #out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

