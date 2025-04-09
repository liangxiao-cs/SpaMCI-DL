import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
from torch_geometric.nn import GATv2Conv
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


# class GCN(Module):
#     def __init__(self, nfeat, nhid, out, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, out)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return x


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


# class GATv2Model(Module):
#     def __init__(self, nfeat, nhid, out, dropout):
#         super(GATv2Model, self).__init__()
#         self.conv1 = GATv2Conv(nfeat, nhid)
#         self.conv2 = GATv2Conv(nhid, out)
#         # self.conv1 = GATv2Conv(nfeat, out)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x


# class decoder(Module):
#     def __init__(self, nfeat, nhid1, nhid2):
#         super(decoder, self).__init__()
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(nhid2, nhid1),
#             torch.nn.BatchNorm1d(nhid1),
#             torch.nn.ReLU()
#         )
#         self.pi = torch.nn.Linear(nhid1, nfeat)
#         self.disp = torch.nn.Linear(nhid1, nfeat)
#         self.mean = torch.nn.Linear(nhid1, nfeat)
#         self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
#         self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
#
#     def forward(self, emb):
#         x = self.decoder(emb)
#         pi = torch.sigmoid(self.pi(x))
#         disp = self.DispAct(self.disp(x))
#         mean = self.MeanAct(self.mean(x))
#         return [pi, disp, mean]


# class Attention(Module):
#     def __init__(self, in_size, hidden_size=16):
#         super(Attention, self).__init__()
#
#         self.project = nn.Sequential(
#             nn.Linear(in_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1, bias=False)
#         )
#
#     def forward(self, z):
#         w = self.project(z)
#         beta = torch.softmax(w, dim=1)
#         return (beta * z).sum(1), beta


# class Attention(Module):
#     def __init__(self, in_feat, out_feat=16):
#         super(Attention, self).__init__()
#         self.in_feat = in_feat
#         self.out_feat = out_feat
#
#         self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
#         self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.w_omega)
#         torch.nn.init.xavier_uniform_(self.u_omega)
#
#     def forward(self, emb1, emb2):
#         emb = []
#         emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
#         emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
#         self.emb = torch.cat(emb, dim=1)
#
#         self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
#         self.vu = torch.matmul(self.v, self.u_omega)
#         self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)
#
#         emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
#
#         return torch.squeeze(emb_combined), self.alpha


# class gg(Module):
#     def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
#         super(gg, self).__init__()
#         self.in_feat = in_feat
#         self.out_feat = out_feat
#         self.dropout = dropout
#         self.act = act
#
#         self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, feat, adj):
#         x = torch.mm(feat, self.weight)
#         x = torch.spmm(adj, x)
#
#         return x

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


class spaLJP(Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(spaLJP, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        # self.BGCN = GCN(nfeat, nhid1, nhid2, dropout)
        # self.GCN_decoder = GCN(nhid2, nhid1, nfeat, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.ReLU(),
            nn.Linear(nhid1, nfeat)
        )
        self.ZINB = zinb_decoder(nfeat, nhid1, nhid2)
        # self.dropout = dropout
        # self.att = Attention(nhid2)
        # self.MLP = nn.Sequential(
        #     nn.Linear(nhid2, nhid2)
        # )
        # self.enc_mask_token = nn.Parameter(torch.zeros(1, nfeat))

    # def encoding_mask_noise(self, x, mask_rate=0.3):
    #     num_nodes = x.shape[0]
    #     perm = torch.randperm(num_nodes, device=x.device)
    #     # random masking
    #     num_mask_nodes = int(mask_rate * num_nodes)
    #     mask_nodes = perm[: num_mask_nodes]
    #     # keep_nodes = perm[num_mask_nodes:]
    #     out_x = x.clone()
    #     # token_nodes = mask_nodes
    #     # out_x[mask_nodes] = 0.0
    #     # out_x[token_nodes] += self.enc_mask_token
    #     out_x[mask_nodes] = self.enc_mask_token
    #     return out_x, mask_nodes

    def forward(self, x, x_bi, sadj):
        # x_mask, mask_nodes = self.encoding_mask_noise(x)
        emb_sp = self.SGCN(x, sadj)
        emb_bi = self.SGCN(x_bi, sadj)  # 将其变成一种约束
        pi, disp, mean = self.ZINB(emb_bi)
        de_emb = self.decoder(emb_sp)
        emb = emb_sp
        # emb_stack = torch.stack([emb_sp, emb_bi], dim=1)
        # emb = torch.stack([emb1, emb2, common_feat, m_feat], dim=1)
        # emb, _ = self.att(emb_stack)
        # emb = self.MLP(emb)
        return emb, emb_bi, de_emb, pi, disp, mean
