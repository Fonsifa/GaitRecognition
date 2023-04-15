import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
sys.path.append("/home/fonsi/桌面/毕设/Gait-Recognition-Using-Smartphones/code/identification/CNN+LSTM/dataset.py")
from dataset import PhoneDataset

#config
model_path = "./cnn_lstm_ckp/"
hidden_size = 1024
length = 128
input_size = 6
num_classes = 118

def onehot(x):
    x_onehot=torch.zeros(x.size(0),118)
    for idx in range(x.size(0)):
        x_onehot[idx,int(x[idx]-0.1)+1]=1
    return x_onehot

#load data
batch_size = 512
train_loader = DataLoader(PhoneDataset("../../data/Dataset#1",train=True),
    batch_size=512,shuffle=True)
test_loader = DataLoader(PhoneDataset("../../data/Dataset#1",train=False),
    batch_size=512,shuffle=False)

# [B,C,H，W]-->[batch_size,1,6，128]

#build Model
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,32,(1,9),stride=(1,2),padding=(0,4))#[B,32,6,64]
        self.pool1 = torch.nn.MaxPool2d((1,2),(1,2))#[B,32,6,32]
        self.conv2 = torch.nn.Conv2d(32,64,(1,3),padding=(0,1))#[B,64,6,32]
        self.conv3 = torch.nn.Conv2d(64,128,(1,3),padding=(0,1))#[B,128,6,32]
        self.pool2 = torch.nn.MaxPool2d((1,2),(1,2))#[B,128,6,16]
        self.conv4 = torch.nn.Conv2d(128,128,(6,1))#[B,128,1,16]
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,num_layers=2)
        self.linear = nn.Linear(hidden_size+2048,num_classes)
    
    def forward(self,x):
        y = x.view(x.size(0),128,6)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.pool1(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(self.pool2(x)))
        x = torch.flatten(x,start_dim=1)#[batch_size,2048]
        z,_ = self.lstm(y)
        z = z[:,-1,:]#[batch_size, hidden_size]
        x = torch.cat([x,z],dim=1)
        x = self.linear(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*********************")
print(device)
net = Net().to(device)
optimizer = optim.Adam(net.parameters(),lr=0.001)

total_correct = 0
acc_best = 0
for epoch in range(200):
    for batch_idx,(x,y) in enumerate(train_loader):
        #x = x.view(x.size(0),128,6)
        x,y=x.to(device),y.to(device)
        output = net(x)
        loss = F.cross_entropy(output,y.squeeze(-1).long().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx%20==0:
            print("epoch :",epoch,"batch_idx :",batch_idx,"loss :", loss.item())
    
    #计算正确率
    total_correct=0
    for _,(x,y) in enumerate(test_loader):
        #x = x.view(x.size(0),128,6)
        x,y = x.to(device),y.to(device)
        y_pred = net(x)
        pred = y_pred.argmax(dim=1)
        correct = pred.eq(y.squeeze(-1)).sum().float().item()
        total_correct+=correct
    acc = total_correct/len(test_loader.dataset)
    print("epoch :",epoch,"acc :",acc)
    if acc>acc_best:
        acc_best = acc
        if os.path.exists(model_path)==False:
            os.mkdir(model_path)
        torch.save(net.state_dict(),model_path+"checkpoint.pth")
