import torch
from torch_sparse import matmul, mul
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

class wAttention(Module):
    def __init__(self):
        super(wAttention, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self,qs,ks,HpGraph_e_k):
        relation = HpGraph_e_k
        qs = qs
        ks = ks
        vs = ks
        attention = torch.matmul(qs,ks.transpose(1,0))
        attention = self.leakyrelu(attention)
        zero_vec = -9e15*torch.ones_like(attention)
        attention = torch.where(relation > 0, attention, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        res =  torch.matmul(attention, vs)
        return res


class sAttention(Module):
    def __init__(self):
        super(sAttention, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self,qs,ks,HpGraph_s_k):
        relation = HpGraph_s_k
        qs = qs
        ks = ks
        vs = ks
        attention = torch.matmul(qs,ks.transpose(1,0))
        attention = self.leakyrelu(attention)
        zero_vec = -9e15*torch.ones_like(attention)
        attention = torch.where(relation > 0, attention, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        res =  torch.matmul(attention, vs)
        return res