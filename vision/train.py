import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from model import *


def test(net, testloader, device):
    total = 0
    correct = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        #labels_list.append(labels)
    
        test = Variable(images.view(images.size(0), 1, 28, 28))
    
        outputs = net(test)
    
        predictions = torch.max(outputs, 1)[1].to(device)
        #predictions_list.append(predictions)
        correct += (predictions == labels).sum()
    
        total += len(labels)
    
    accuracy = correct * 100 / total
    return accuracy

def train(net, num_epochs, trainloader, testloader, criterion, optimizer, device):

    for epoch in range(num_epochs):
    
        for images, labels in trainloader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)
            
            train = Variable(images.view(images.size(0), 1, 28, 28))
            labels = Variable(labels)
            
            # Forward pass 
            outputs = net(train)
            loss = criterion(outputs, labels)
            
            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            
            #Propagating the error backward
            loss.backward()
            
            # Optimizing the parameters
            optimizer.step()
        

        # Testing the model
        acc = test(net, testloader, device)
        print(f'test acc at epoch {epoch} is acc = {acc}')    
            

if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = 'emnist' #emnist or fashion_mnist
    
    
    if dataset == 'fashion_mnist':
        num_classes = 10
    else:
        num_classes = 47
    num_epochs = 10 #fixed
    learning_rate = 0.1 #fixed

    #############
    #net = CentralisedCNN(num_classes=num_classes) # default, not for vfl network
    net = MLP(num_classes=num_classes) # only linear layers
    #net = CNN2(num_classes=num_classes) # 2 CNN then linear layers, second CNN need to change the kernel size to 1
    #net = CNN_and_CNN(num_classes=num_classes) # 1 CNN locally, NO linear layer in between, then CNN globally
    #net = CNN_linear_CNN(num_classes=num_classes) # 1 CNN locally, then liner layer, then CNN globally
    
    net.to(device)
    if dataset == 'fashion_mnist':
        train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  

    else:
        train_set = torchvision.datasets.EMNIST("./data", split = 'balanced', download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.EMNIST("./data", split = 'balanced', download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  
    
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=100)
    testloader = torch.utils.data.DataLoader(test_set,batch_size=100)


    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    print(net)
    
    train(net, num_epochs, trainloader, testloader, criterion, optimizer, device)