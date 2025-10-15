import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
from torch.nn.modules.module import Module


class GCN(Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu_(x)
        x = self.gc2(x, adj)
        return x

class zinb_decoder(Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(zinb_decoder, self).__init__()
        self.z_decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.z_decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return pi, disp, mean


class SpaMCI(Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(SpaMCI, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.ReLU(),
            nn.Linear(nhid1, nfeat)
        )
        self.ZINB = zinb_decoder(nfeat, nhid1, nhid2)

    def forward(self, x, x_bi, sadj):
        emb_sp = self.SGCN(x, sadj)
        emb_bi = self.SGCN(x_bi, sadj)
        pi, disp, mean = self.ZINB(emb_bi)
        de_emb = self.decoder(emb_sp)
        emb = emb_sp
        return emb, emb_bi, de_emb, pi, disp, mean
