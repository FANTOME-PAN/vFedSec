import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

#from data import *
from model import *

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]


def test(net, testloader):
    total = 0
    correct = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        #labels_list.append(labels)
    
        test = Variable(images.view(100, 1, 28, 28))
    
        outputs = net(test)
    
        predictions = torch.max(outputs, 1)[1].to(device)
        #predictions_list.append(predictions)
        correct += (predictions == labels).sum()
    
        total += len(labels)
    
    accuracy = correct * 100 / total
    return accuracy

def train(net, num_epochs, trainloader, testloader, criterion, optimizer):
    count = 0
    # Lists for visualization of loss and accuracy 
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
    
        for images, labels in trainloader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)
        
            train = Variable(images.view(100, 1, 28, 28))
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
        
            count += 1

        # Testing the model
        acc = test(net, testloader)
        print(f'test acc at epoch {epoch} is acc = {acc}')    
            

if __name__ =='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #net = FashionCNN() # default, not for vfl network
    #net = CNN1() #1 CNN then linear layers
    #net = MLP() # only linear layers
    #net = CNN2() # 2 CNN then linear layers, second CNN need to change the kernel size to 1
    #net = CNN_and_CNN() # 1 CNN locally, NO linear layer in between, then CNN globally
    net = CNN_linear_CNN() # 1 CNN locally, then liner layer, then CNN globally
    
    net.to(device)
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=100)
    testloader = torch.utils.data.DataLoader(test_set,batch_size=100)


    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(net)

    num_epochs = 10
    
    train(net, num_epochs, trainloader, testloader, criterion, optimizer)