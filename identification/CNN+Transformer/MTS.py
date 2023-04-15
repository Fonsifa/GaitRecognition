from typing import Any
import torch
from torch import nn, Tensor,optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import sys
sys.path.append("/home/fonsi/桌面/毕设/Gait-Recognition-Using-Smartphones/code/identification/ViT/dataset.py")
from dataset import PhoneDataset
from transformer import *
data_path = "../../data/Dataset#2"
model_path = "./trans_ckp/"
num_classes = 20
    
class TST(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.cnn = CNNBlock()
        self.tst = TSTransformerEncoderClassiregressor(feat_dim=6,max_len=128,d_model=32,
                                          n_heads=4,num_layers=4,dim_feedforward=256,pos_encoding="learnable",dropout=0.,
                                          num_classes=num_classes
                                          )
        self.linear = nn.Linear(32*128,num_classes)
    def forward(self,x):#x batch,128,6
        y = x
        # x = x.unsqueeze(-1).permute(0,3,2,1)
        # x = self.cnn(x)
        # x = torch.flatten(x,start_dim=1)#[batch_size,2048]
        y = self.tst(y, torch.ones(y.size(0),128).bool().cuda())
        #output = self.linear(torch.cat([x,y],dim=1))
        output = self.linear(y)
        return output

class CNNTST(nn.Module):
    def __init__(self,cnn_fix=False,trans_fix=False) -> None:
        super().__init__()
        self.cnn = CNNBlock()
        if cnn_fix:
            self.cnn.load_state_dict(torch.load("./fix_model/cnn_block_20.pth"))
        self.cnn_fix = cnn_fix
        self.tst = TSTransformerEncoderClassiregressor(feat_dim=6,max_len=128,d_model=32,
                                          n_heads=4,num_layers=4,dim_feedforward=256,pos_encoding="learnable",dropout=0.,
                                          num_classes=num_classes
                                          )
        if trans_fix:
            self.tst.load_state_dict(torch.load("./fix_model/trans_fix_20.pth"))
        self.trans_fix = trans_fix
        self.linear = nn.Linear(32*128+2048,num_classes)
    def forward(self,x):#x batch,128,6
        y = x
        x = x.unsqueeze(-1).permute(0,3,2,1)
        if self.cnn_fix:
            with torch.no_grad():
                x = self.cnn(x)
        else :
            x = self.cnn(x)
        x = torch.flatten(x,start_dim=1)#[batch_size,2048]
        if self.trans_fix:
            with torch.no_grad():
                y = self.tst(y, torch.ones(y.size(0),128).bool().cuda())
        else:
            y = self.tst(y, torch.ones(y.size(0),128).bool().cuda())
        output = self.linear(torch.cat([x,y],dim=1))
        return output

#load data
batch_size = 256
train_loader = DataLoader(PhoneDataset(data_path,train=True),
    batch_size=batch_size,shuffle=True)
test_loader = DataLoader(PhoneDataset(data_path,train=False),
    batch_size=batch_size,shuffle=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("*********************")
print(device)
net = CNNTST(trans_fix=True).to(device)
optimizer = optim.Adam(net.parameters(),lr=0.00025)

total_correct = 0
acc_best = 0
for epoch in range(300):
    for batch_idx,(x,y) in enumerate(train_loader):
        # x = x.view(x.size(0),1,6,128)
        x,y=x.permute(0,3,2,1).squeeze(-1).to(device),y.to(device)
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
        x,y = x.permute(0,3,2,1).squeeze(-1).to(device),y.to(device)
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
        torch.save(net.tst.state_dict(),model_path+"trans_checkpoint.pth")
