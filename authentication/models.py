import torch
from torch import nn
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

import sys
from typing import Optional, Any
import math


class CNNBlock(nn.Module):
    def __init__(self,vertical=False,tcn=False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,32,(1,9),stride=(1,2),padding=(0,4)),#[B,32,6,64]
            nn.ReLU(),
            nn.MaxPool2d((1,2),(1,2)),#[B,32,6,32]
            nn.Conv2d(32,64,(1,3),padding=(0,1)),#[B,64,6,32]
            nn.ReLU(),
            nn.Conv2d(64,128,(1,3),padding=(0,1)),#[B,128,6,32]
            nn.ReLU(),
            nn.MaxPool2d((1,2),(1,2)),#[B,128,6,16]
            nn.Conv2d(128,128,(6,1)),#[B,128,1,16]
            nn.ReLU()
            )
        if vertical:
            self.block = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,32,(1,9),stride=(1,2),padding=(0,4)),#[B,32,6,64]
            nn.ReLU(),
            nn.MaxPool2d((1,2),(1,2)),#[B,32,6,32]
            nn.Conv2d(32,64,(1,3),padding=(0,1)),#[B,64,6,32]
            nn.ReLU(),
            nn.Conv2d(64,128,(1,3),padding=(0,1)),#[B,128,6,32]
            nn.ReLU(),
            nn.MaxPool2d((1,2),(1,2)),#[B,128,6,16]
            nn.Conv2d(128,128,(11,1)),#[B,128,1,16]
            nn.ReLU()
            )
        

    
    def forward(self,x):
        return self.block(x)

########################################################################################################

class SiaCNNAuth(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn_block = CNNBlock(tcn=True)
        self.linear1 = nn.Linear(2048,512)
        self.linear2 = nn.Linear(512,32)

    def forward_once(self,x):
        x = x.reshape(-1,1,6,128)
        x = self.cnn_block(x)
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def forward(self,x1,x2):
        output1,output2 = self.forward_once(x1),self.forward_once(x2)
        #output = abs(self.forward_once(x1)-self.forward_once(x2))
        return output1,output2
    
class OriginCNNAuth(nn.Module):
    def __init__(self,vertical=False) -> None:
        super().__init__()
        self.vertical = vertical
        self.cnn_block = CNNBlock(vertical,tcn=True)
        self.linear1 = torch.nn.Linear(2048*2,2)
    
    def forward(self,x1,x2):
        x1 = x1.reshape(-1,1,6,128)
        x2 = x2.reshape(-1,1,6,128)
        if self.vertical:
            x = torch.cat([x1,x2],dim=2)
        else:
            x = torch.cat([x1,x2],dim=3)

        output = self.cnn_block(x)
        x = torch.flatten(output,start_dim=1)
        x = self.linear1(x)
        return x
    
#########################################################################################################

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class SiaCNNLSTMAuth(nn.Module):
    def __init__(self,input_size=6,hidden_size=1024) -> None:
        super().__init__()
        self.cnn_block = CNNBlock()
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,num_layers=2)
        self.linear = nn.Linear(hidden_size+2048,32)
    
    def forward_once(self,x):
        y = x.permute(0,2,1)
        x = x.reshape(-1,1,6,128)
        x = self.cnn_block(x)
        x = torch.flatten(x,start_dim=1)#[batch_size,2048]
        z,_ = self.lstm(y)
        z = z[:,-1,:]#[batch_size, hidden_size]
        x = torch.cat([x,z],dim=1)
        x = self.linear(x)
        return x
    
    def forward(self,x1,x2):
        return self.forward_once(x1),self.forward_once(x2)

class SiaCNNLSTMAuthSer(nn.Module):
    def __init__(self,input_size=128,hidden_size=1024) -> None:
        super().__init__()
        self.cnn_block = CNNBlock()
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,num_layers=2)
        self.linear = nn.Linear(hidden_size,32)
    
    def forward_once(self,x):
        x = x.reshape(-1,1,6,128)
        x = self.cnn_block(x)#b,128,1,16
        x = x.permute(0,3,1,2).squeeze(-1)
        z,_ = self.lstm(x)
        z = z[:,-1,:]#[batch_size, hidden_size]
        x = self.linear(z)
        return x
    
    def forward(self,x1,x2):
        return self.forward_once(x1),self.forward_once(x2)
    


class OriginCNNLSTMAuth(nn.Module):
    def __init__(self,vertical=False,hidden_size=1024) -> None:
        super().__init__()
        self.vertical = vertical
        self.cnn_block = CNNBlock()
        if vertical:
            input_size=256
        else :
            input_size=128
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,num_layers=2)
        self.linear = nn.Linear(hidden_size,256)
        self.linear2 = nn.Linear(256,2)
    
    def forward_once(self,x):
        x = x.reshape(-1,1,6,128)
        x = self.cnn_block(x)
        return x
    
    def forward(self,x1,x2):
        x1 = self.forward_once(x1).permute(0,3,1,2).squeeze(-1)
        x2 = self.forward_once(x2).permute(0,3,1,2).squeeze(-1)
        if self.vertical:
            x = torch.cat([x1,x2],dim=2)
        else :
            x = torch.cat([x1,x2],dim=1)
        z,_ = self.lstm(x)
        z = z[:,-1,:]
        x = torch.sigmoid(self.linear(z))
        return self.linear2(x)
    
class CNNLSTMAuthPara(nn.Module):
    def __init__(self,vertical=False,hidden_size=1024) -> None:
        super().__init__()
        self.vertical = vertical
        self.cnn_block = CNNBlock(vertical)
        if vertical:
            input_size=12
        else :
            input_size=6
        self.lstm = nn.LSTM(input_size,hidden_size,batch_first=True,num_layers=2)
        self.linear = nn.Linear(hidden_size+2048*2,512)
        self.linear2 = nn.Linear(512,2)
    
    def forward(self,x1,x2):
        x1 = x1.reshape(-1,1,6,128)
        x2 = x2.reshape(-1,1,6,128)
        if self.vertical:
            x = torch.cat([x1,x2],dim=2)
        else:
            x = torch.cat([x1,x2],dim=3)
        output1 = self.cnn_block(x)
        output1 = torch.flatten(output1,start_dim=1)
        output2,_ = self.lstm(x.permute(0,3,2,1).squeeze(-1))
        output2 = F.relu(output2[:,-1,:])
        x = torch.cat([output1,output2],dim=1)
        x = F.relu(self.linear(x))
        return self.linear2(x)
        
########################################################################################################################


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks) 

        # permute because pytorchself.max_len_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        return output
        
    
class SiaCNNTransAuth(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn_block = CNNBlock()
        self.tst = TSTransformerEncoderClassiregressor(feat_dim=6,max_len=128,d_model=16,
                                          n_heads=2,num_layers=2,dim_feedforward=128,
                                          num_classes=118,dropout=0.,pos_encoding="learnable",
                                          )
        self.linear = nn.Linear(16*128+2048,32)
    def forward_once(self,x):
        y = x.permute(0,2,1)#batch,128,6
        x = x.unsqueeze(-1).permute(0,3,1,2)
        x = self.cnn_block(x)
        x = torch.flatten(x,start_dim=1)#[batch_size,2048]
        y = self.tst(y, torch.ones(y.size(0),128).bool().cuda())
        output = self.linear(torch.cat([x,y],dim=1))
        return output
    
    def forward(self,x1,x2):
        return self.forward_once(x1),self.forward_once(x2)
    
class SiaCNNTransAuthSer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn_block = CNNBlock()
        self.tst = TSTransformerEncoderClassiregressor(feat_dim=16,max_len=128,d_model=64,
                                          n_heads=4,num_layers=2,dim_feedforward=256,dropout=0.,pos_encoding="learnable",
                                          num_classes=118
                                          )
        self.linear = nn.Linear(64*128,32)
    def forward_once(self, x):
        x = x.reshape(-1,1,6,128)
        x = self.cnn_block(x)#b,128,1,16
        x = x.permute(0,3,1,2).squeeze(-1)
        x = F.relu(self.tst(x,torch.ones(x.size(0),128).bool().cuda()))
        x = self.linear(x)
        return x
    def forward(self,x1,x2):
        return self.forward_once(x1),self.forward_once(x2)




class CNNTransAuthSer(nn.Module):
    def __init__(self,vertical=False) -> None:
        super().__init__()
        self.vertical = vertical
        self.cnn_block = CNNBlock()
        if vertical:
            self.feat_dim = 32
            self.max_len = 128
        else:
            self.feat_dim = 16
            self.max_len = 256
        self.tst = TSTransformerEncoderClassiregressor(feat_dim=self.feat_dim,max_len=self.max_len,d_model=64,
                                          n_heads=4,num_layers=2,dim_feedforward=256,pos_encoding="learnable",norm="LayerNorm",dropout=0.,
                                          num_classes=118
                                          )
        self.linear = nn.Linear(64*self.max_len,1024)
        self.linear2 = nn.Linear(1024,2)
        
    def forward_once(self,x):
        x = x.reshape(-1,1,6,128)
        x = self.cnn_block(x)
        return x


    def forward(self,x1,x2):
        x1 = self.forward_once(x1).permute(0,1,3,2).squeeze(-1)
        x2 = self.forward_once(x2).permute(0,1,3,2).squeeze(-1)

        if self.vertical:
            x = torch.cat([x1,x2],dim=2)
        else :
            x = torch.cat([x1,x2],dim=1)

        x = self.tst(x, torch.ones(x.size(0),self.max_len).bool().cuda())
        x = F.sigmoid(self.linear(x))
        return self.linear2(x)
         

class CNNTransAuthPara(nn.Module):
    def __init__(self,vertical=False) -> None:
        super().__init__()
        self.vertical = vertical
        self.cnn_block = CNNBlock(vertical)
        if vertical:
            self.feat_dim = 12
            self.max_len = 128
        else:
            self.feat_dim = 6
            self.max_len = 256
        self.tst = TSTransformerEncoderClassiregressor(feat_dim=self.feat_dim,max_len=self.max_len,d_model=32,
                                          n_heads=2,num_layers=2,dim_feedforward=256,pos_encoding="learnable",
                                          num_classes=118,norm="LayerNorm",dropout=0.1
                                          )
        self.linear = nn.Linear(32*self.max_len+2048*2,512)
        self.linear2 = nn.Linear(512,2)
    
    def forward(self,x1,x2):
        x1 = x1.reshape(-1,1,6,128)
        x2 = x2.reshape(-1,1,6,128)
        if self.vertical:
            x = torch.cat([x1,x2],dim=2)
        else:
            x = torch.cat([x1,x2],dim=3)

        output = self.cnn_block(x)
        x1 = torch.flatten(output,start_dim=1)
        x2 = self.tst(x.permute(0,3,2,1).squeeze(-1), torch.ones(x.size(0),self.max_len).bool().cuda())
        output = torch.cat([x1,x2],dim=1)
        output = F.sigmoid(self.linear(output))
        output = self.linear2(output)
        return output

class TransAuth1(nn.Module):
    def __init__(self,vertical=False) -> None:
        super().__init__()
        self.vertical = vertical
        self.cnn_block = CNNBlock(vertical,True)
        if vertical:
            self.feat_dim = 12
            self.max_len = 128
        else:
            self.feat_dim = 6
            self.max_len = 256
        self.tst = TSTransformerEncoderClassiregressor(feat_dim=self.feat_dim,max_len=self.max_len,d_model=32,
                                          n_heads=2,num_layers=2,dim_feedforward=256,pos_encoding="learnable",dropout=0,
                                          num_classes=118,norm="LayerNorm"
                                          )
        self.linear = nn.Linear(32*self.max_len,2)
        #self.linear2 = nn.Linear(32,2)
    
    def forward(self,x1,x2):
        x1 = x1.reshape(-1,1,6,128)
        x2 = x2.reshape(-1,1,6,128)
        if self.vertical:
            x = torch.cat([x1,x2],dim=2)
        else:
            x = torch.cat([x1,x2],dim=3)

        output = self.tst(x.permute(0,3,2,1).squeeze(-1), torch.ones(x.size(0),self.max_len).bool().cuda())
        #output = F.sigmoid(self.linear(output))
        output = self.linear(output)
        return output
#######################################################################################################################
