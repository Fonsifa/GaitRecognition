from torch.nn import Module
from torch import nn
import torch
import numpy as np
import math
import torch.nn.functional as F


class SelfAttentionConv(Module):
    def __init__(self, k,appendix="channel" , headers = 8, kernel_size = 3, mask_next = True, mask_diag = False):
        super().__init__()
        
        self.appendix=appendix
        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag
        
        h = headers
        
        # Query, Key and Value Transformations
        
        padding = (kernel_size-1)
        self.padding_opertor = nn.ConstantPad1d((padding,0), 0)
        
        self.toqueries = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tokeys = nn.Conv1d(k, k*h, kernel_size, padding=0 ,bias=True)
        self.tovalues = nn.Conv1d(k, k*h, kernel_size = 1 , padding=0 ,bias=False) # No convolution operated
        
        # Heads unifier
        self.unifyheads = nn.Linear(k*h, k)
    def forward(self, x,_):
        
        # Extraction dimensions
        b, t, k  = x.size() # batch_size, number_of_timesteps, number_of_time_series
        
        
        # Checking Embedding dimension
        assert self.k == k, 'Number of time series '+str(k)+' didn t much the number of k '+str(self.k)+' in the initiaalization of the attention layer.'
        h = self.headers
        
        #  Transpose to see the different time series as different channels
        x = x.transpose(1,2)
        x_padded = self.padding_opertor(x)
        
        # Query, Key and Value Transformations
        queries = self.toqueries(x_padded).view(b,k,h,t)
        keys = self.tokeys(x_padded).view(b,k,h,t)
        values = self.tovalues(x).view(b,k,h,t)
        
        # Transposition to return the canonical format
        queries = queries.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        queries = queries.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        values = values.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        values = values.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        keys = keys.transpose(1,2) # batch, header, time serie, time step (b, h, k, t)
        keys = keys.transpose(2,3) # batch, header, time step, time serie (b, h, t, k)
        
        
        # Weights 
        queries = queries/(k**(.25))
        keys = keys/(k**(.25))
        
        queries = queries.transpose(1,2).contiguous().view(b*h, t, k)
        keys = keys.transpose(1,2).contiguous().view(b*h, t, k)
        values = values.transpose(1,2).contiguous().view(b*h, t, k)
        
        weights = torch.bmm(queries, keys.transpose(1,2))
        #np.save(self.appendix+"socres.npy",weights.cpu().detach().numpy())
        
        ## Mask the upper & diag of the attention matrix
        if self.mask_next :
            if self.mask_diag :
                indices = torch.triu_indices(t ,t , offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
            else :
                indices = torch.triu_indices(t ,t , offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')
        
        # Softmax 
        weights = F.softmax(weights, dim=2)
        
        # Output
        output = torch.bmm(weights, values)
        output = output.view(b,h,t,k)
        output = output.transpose(1,2).contiguous().view(b,t, k*h)
        
        return self.unifyheads(output),None # shape (b,t,k)


# class MultiHeadAttention(Module):
#     def __init__(self,
#                  d_model: int,
#                  q: int,
#                  v: int,
#                  h: int,
#                  device: str,
#                  mask: bool=False,
#                  dropout: float = 0.1):
#         super(MultiHeadAttention, self).__init__()

#         self.W_q = torch.nn.Linear(d_model, q * h)
#         self.W_k = torch.nn.Linear(d_model, q * h)
#         self.W_v = torch.nn.Linear(d_model, v * h)

#         self.W_o = torch.nn.Linear(v * h, d_model)

#         self.device = device
#         self._h = h
#         self._q = q

#         self.mask = mask
#         self.dropout = torch.nn.Dropout(p=dropout)
#         self.score = None

#     def forward(self, x, stage):
#         Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
#         K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
#         V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

#         score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
#         self.score = score

#         if self.mask and stage == 'train':
#             mask = torch.ones_like(score[0])
#             mask = torch.tril(mask, diagonal=0)
#             score = torch.where(mask > 0, score, torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))

#         score = F.softmax(score, dim=-1)

#         attention = torch.matmul(score, V)

#         attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

#         self_attention = self.W_o(attention_heads)

#         return self_attention,score
