from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtWidgets import QComboBox
import glob
import torch
import torch.nn as nn
from torch import  optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from torchsummary import summary
import cv2
import torch.nn.functional as F

Label_Cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x) 
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.classifier(out)
        # print(out.shape)
        return out



class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('HW1_5.ui', self)                    
        self.TrainImgBtn.clicked.connect(self.LoadImage) 
        self.HypeParaBtn.clicked.connect(self.ShowHyperparameter)
        self.ModelStructBtn.clicked.connect(self.modelSummary)
        self.AccuracyBtn.clicked.connect(self.showGraph)
        self.TestImgBtn.clicked.connect(self.showResult)
        self.show()
 
    def Parameter_Training(self):
        Batch_size = 100
        Epochs = 50
        Learning_Rate = 0.001
        return Epochs, Batch_size, Learning_Rate
       
    def PreTraining(self):
        EPOCHS , Batch_size, LR = self.Parameter_Training()
        train_dataset = datasets.CIFAR10('./data', train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(train_dataset, batch_size=Batch_size , shuffle=True)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(test_dataset, batch_size=Batch_size , shuffle=False)
        return train_dataset, train_loader, test_dataset, test_loader

    def LoadImage(self):
        train_dataset, trainloader, testset, testloader = self.PreTraining()
        fig, axes = plt.subplots(1, 10, figsize=(12,5))
        text = ''
        for i in range(10):
            index = random.randint(0,9999)
            image = train_dataset[index][0] 
            label = train_dataset[index][1] 
            # text += "The picture %s is showing: The %s\n" %(str(i+1),Label_Cifar10[int(label)])
            Showimage = image.numpy()
            Showimage = np.transpose(Showimage, (1, 2, 0))
            Showlabel = Label_Cifar10[int(label)]
    
            axes[i].imshow(Showimage)
            axes[i].set_xlabel(Showlabel)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        print(text)
        plt.show()

    def findOptimizer(self):
        EPOCHS , BATCH_SIZE, LR = self.Parameter_Training()
        net = self.findUsedNet()
        OPTIMIZER = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        return OPTIMIZER

    def ShowHyperparameter(self):
        EPOCHS , BATCH_SIZE, LR = self.Parameter_Training()
        optimizer = self.findOptimizer()
        print("Hyperparameter:")
        print("Batch size: ", BATCH_SIZE)
        print("Learning rate: ", LR)
        print("Optimizer: ", optimizer.__class__.__name__)    
    

    def findUsedNet(self):
        device = self.getDevice()
        NET= VGG16().to(device)
        return NET

    def getDevice(self):
        # GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print('GPU state:', device)
        return device

    def modelSummary(self):
        device = self.getDevice()
        net = self.findUsedNet()
        summary(net, (3, 32, 32))
    
    def showGraph(self):
        img = cv2.imread('./Graph.jpg')
        cv2.imshow('Graph',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    def showResult(self):
        PATH = "./trained_Vgg16.pth"
        device = self.getDevice()
        net = VGG16().to(device)
        net.load_state_dict(torch.load(PATH))
        net.eval()

        index = int(self.NumOfVal.text())
        train_dataset, train_loader, test_dataset, test_loader = self.PreTraining()
        image1 = test_dataset[index][0] 
        label = test_dataset[index][1] 


        image = image1.to(device).unsqueeze(0)
        output = net(image)
        _, pred = torch.max(output, 1)
        output = F.softmax(output, dim=1)
        predictList = [Element.item() for Element in output.flatten()]

        plt.figure(figsize=(15, 3))
        plt.subplot(1, 2, 1)
        imgTranposed = np.transpose(image1, (1, 2, 0))
        imgClipped = np.clip(imgTranposed, 0, 1)
        plt.imshow(imgClipped)

        plt.subplot(1, 2, 2)
        cifarLen = np.arange(len(Label_Cifar10))
        plt.bar(cifarLen, predictList)
        plt.xticks(cifarLen, Label_Cifar10)
        plt.show()

        text = "Picture %4s is: %5s\nRight answer is: %5s" %(str(index),Label_Cifar10[pred.item()],Label_Cifar10[int(label)])
        print(text)
  

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()