import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
sys.path.append("/home/fonsi/桌面/毕设/Gait-Recognition-Using-Smartphones/code/identification/LSTMs/dataset.py")
from dataset import PhoneDataset

#config
data_path = "../../data/Dataset#1"
model_path = "./lstm_ckp/"
hidden_size = 64
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
train_loader = DataLoader(PhoneDataset(data_path,train=True),
    batch_size=512,shuffle=True)
test_loader = DataLoader(PhoneDataset(data_path,train=False),
    batch_size=512,shuffle=False)

# [B,C,H，W]-->[batch_size,1,6，128]

#build Model
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.bn = nn.BatchNorm2d(1)
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,bidirectional=False,num_layers=2,
                            dropout =0.2)
        self.linear = nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        #x = self.bn(x)
        x = x.view(x.size(0),128,6)
        x,_ = self.lstm(x)
        x = x[:,-1,:]
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