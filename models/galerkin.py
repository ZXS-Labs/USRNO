import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from utils import show_feature_map

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

@register('galerkin')
class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.qkv_proj.apply(init_weights)
        #self.w = nn.init.kaiming_normal_(nn.Parameter(torch.randn(self.heads, self.headc, 3*self.headc)))
        #self.b = nn.init.kaiming_normal_(nn.Parameter(torch.randn(self.heads, 3*self.headc)))
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        #self.kln = LayerNorm((self.headc))
        #self.vln = LayerNorm((self.headc))

        #self.kin = nn.ModuleList([nn.InstanceNorm1d(self.headc) for _ in range(self.heads)])
        #self.vin = nn.ModuleList([nn.InstanceNorm1d(self.headc) for _ in range(self.heads)])

        self.act = nn.GELU()
    
    def forward(self, x, name='0'):
        B, C, H, W= x.shape
        bias = x

        #if name == 0: show_feature_map((x[:,0:10:1]),'out/edsr','feat')
        
        #x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        #qkv = x.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).reshape(B, H*W, self.heads, 3*self.headc)
        #qkv = qkv.reshape(B, H*W, self.heads, 3*self.headc)

        #x = x.permute(0, 2, 3, 1).reshape(B, H*W, self.heads, self.headc)
        #qkv = torch.einsum('bnhi,hio->bnho', x, self.w) + self.b
        
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        #ranksq = torch.linalg.matrix_rank(q.squeeze(0), tol = 1e-2)
        #ranksk = torch.linalg.matrix_rank(k.squeeze(0), tol = 1e-2)
        #ranksvo = torch.linalg.matrix_rank(v.squeeze(0), tol = 1e-2)
        #show_feature_map((k).permute(0,3,1,2).reshape(B,C,H,W)[:,0:10:1],'k',name)
        #show_feature_map((v),'v',name)
        #show_feature_map((q),'q',name)
        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        #show_feature_map(v,'out/edsr/kv',name=name)
        #rankskv = torch.linalg.matrix_rank(v.squeeze(0), tol = 1e-2)
        v = torch.matmul(q, v)

        #ranksv = torch.linalg.matrix_rank(v.squeeze(0), tol = 1e-2)
        #print("rankq, rankk, rankvo rankv, rankkv:",ranksq,ranksk,ranksvo,ranksv,rankskv)

        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
        #show_feature_map((v.permute(0,3,1,2)[:,0:10:1]),'z',name)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias

def init_weights(m, delta = 0.01):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight, gain = 1.0e-4) + (delta * torch.diag(torch.ones(
                m.weight.size(0), m.weight.size(1), dtype=torch.float32
            )).unsqueeze(-1).unsqueeze(-1))