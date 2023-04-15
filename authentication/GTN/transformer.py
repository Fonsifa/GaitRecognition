from torch.nn import Module
from torch import nn
import torch
from torch.nn import ModuleList
from encoder import Encoder
import math
import torch.nn.functional as F

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).permute(1,0,2)

class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 vertical: bool=False,
                 pe: bool = True,
                 mask: bool = False):
        super(Transformer, self).__init__()
        if not vertical:
            d_input=d_input*2
        else :
            d_channel=d_channel*2
        self.vertical = vertical
        d_model1=d_model*2
        d_hidden1=d_hidden*1
        q1=q*2
        v1=v*2
        h1=h*1

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  conv=True,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model1,
                                                  d_hidden=d_hidden1,
                                                  q=q1,
                                                  v=v1,
                                                  h=h1,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model1)
        # self.embedding_input1 = torch.nn.Conv1d(d_channel,d_channel,5,4,2)

        self.gate = torch.nn.Linear(d_model * d_input + d_model1 * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model1 * d_channel, 256)
        self.output_linear1 = nn.Linear(256,2)

        self.pe = LearnablePositionalEncoding(d_model,max_len=d_input)
        #self.pe1 = LearnablePositionalEncoding(d_model,max_len=d_input)
        #if vertical:
        #    self.pe1 = LearnablePositionalEncoding(d_model,max_len=d_channel)
        #else:
        #    self.pe1=None
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x1, x2, stage):
        """
        前向传播
        :param x: 输入
        :param stage: 用于描述此时是训练集的训练过程还是测试集的测试过程  测试过程中均不在加mask机制
        :return: 输出，gate之后的二维向量，step-wise encoder中的score矩阵，channel-wise encoder中的score矩阵，step-wise embedding后的三维矩阵，channel-wise embedding后的三维矩阵，gate
        """
        # step-wise
        # score矩阵为 input， 默认加mask 和 pe
        if not self.vertical:
            x = torch.cat([x1.permute(0,2,1),x2.permute(0,2,1)],dim=1)
        else:
            x = torch.cat([x1.permute(0,2,1),x2.permute(0,2,1)],dim=2)
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1 

        encoding_1 = self.pe(encoding_1)

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # score矩阵为channel 默认不加mask和pe
        encoding_2 = self.embedding_input(x.transpose(-1,-2))
        channel_to_gather = encoding_2

        #encoding_2 = self.pe1(encoding_2)

        #if self.vertical:
        #    encoding_2 = self.pe1(encoding_2)

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # 三维变二维
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # 输出
        output = torch.sigmoid(self.output_linear(encoding))
        output = self.output_linear1(output)

        return output#, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate

