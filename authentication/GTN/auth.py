import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import os
import sys
sys.path.append("/home/fonsi/桌面/毕设/Gait-Recognition-Using-Smartphones/code/authentication")
from dataset import PhoneDataset
from transformer import *

data_path = "/root/autodl-tmp/Gait-Recognition-Using-Smartphones/code/data/Dataset#5"
model_path = "./cnn_ckp/"


#load data
batch_size = 512
train_loader = DataLoader(PhoneDataset(data_path,train=True),
    batch_size=batch_size,shuffle=True)
test_loader = DataLoader(PhoneDataset(data_path,train=False),
    batch_size=batch_size,shuffle=False)

#build model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*********************")
print(device)
net = Transformer(d_model=64,d_input=128,d_channel=6,d_hidden=256,d_output=2,q=16,v=16,h=2,vertical=True,N=2,dropout=0.05,mask=True,device=device).to(device)
optimizer = optim.Adam(net.parameters(),lr = 0.001)
loss_func = torch.nn.CrossEntropyLoss()

total_correct = 0
acc_best = 0
losses=[]
accs=[]

for epoch in range(300):
    for batch_idx,(x1,x2,y) in enumerate(train_loader):
        # x = x.view(x.size(0),1,6,128)
        x1,x2,y=x1.to(device),x2.to(device),y.to(device)
        output = net(x1,x2,'train')
        loss = loss_func(output,y.squeeze(-1).long().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx%20==0:
            print("epoch :",epoch,"batch_idx :",batch_idx,"loss :", loss.item())
    
    #计算正确率
    total_correct=0
    y_true=np.array([])
    y_socres=np.array([])
    for idx,(x1,x2,y) in enumerate(test_loader):
        x1,x2,y=x1.to(device),x2.to(device),y.to(device)
        y_pred = net(x1,x2,'test')
        if idx==0:
            y_scores=y_pred[:,1].cpu().detach().numpy()
            y_true=y.unsqueeze(-1).cpu().detach().numpy()
        else :
            y_scores=np.concatenate((y_scores,y_pred[:,1].cpu().detach().numpy()))
            y_true=np.concatenate((y_true,y.unsqueeze(-1).cpu().detach().numpy()))

        pred = y_pred.argmax(dim=1)
        correct = pred.eq(y.squeeze(-1)).sum().float().item()
        total_correct+=correct
    acc = total_correct/len(test_loader.dataset)
    accs.append(acc)
    print("epoch :",epoch,"acc :",acc)
    np.savez("./train.npz",np.array(losses),np.array(accs))
    if acc>acc_best:
        fpr,tpr,tho =metrics.roc_curve(y_true,y_scores,pos_label=1)
        # plt.plot(fpr,tpr,marker = 'o')
        # plt.show()
        auc = metrics.auc(fpr,tpr)

        np.savez('ConvGTN_V.npz',fpr,tpr)
        print(auc)
        acc_best = acc
        if os.path.exists(model_path)==False:
            os.mkdir(model_path)
        torch.save(net.state_dict(),model_path+"checkpoint.pth")
