import torch.optim as optim
from utils import *
from models import SpaMCI
import os
from config import Config
from sklearn import metrics
import pandas as pd
import random
from torch.backends import cudnn
import torch.nn.functional as F
import torch.nn as nn
import ot
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class NB():
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def normalize_adj(adj, protocol='10X'):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    if protocol in ['Stereo-seq', 'Slide-seqV2']:
        return sparse_mx_to_torch_sparse_tensor(adj)
    else:
        return torch.FloatTensor(adj.toarray())


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']

    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)
    return new_type


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_file = 'Human_Breast_Cancer.ini'
    config = Config(config_file)
    file_fold = '/home/liangxiao/lllxxx/Human_Breast_Cancer/'
    adata = sc.read_visium(file_fold)
    print(adata)
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata.obs['ground_truth'] = df_meta_layer.values
    n_clusters = len(adata.obs['ground_truth'].unique())
    adata.var_names_make_unique()
    preprocess(adata)
    adata = adata[:, adata.var['highly_variable']]

    # print(adata.X)

    # location = (adata.obsm['spatial']).toarray()
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        feat_X = adata.X.toarray()
        feat_X_bi = ((adata.X > 0).astype(np.float64)).toarray()
    else:
        feat_X = adata.X
        feat_X_bi = (adata.X > 0).astype(np.float64)
    sadj = spatial_construct_graph1(adata, n_neighbors=6)
    num_sadj = torch.FloatTensor(sadj)
    sadj = normalize_adj(sadj.copy(), protocol='10X')
    features = torch.FloatTensor(feat_X)
    features_bi = torch.FloatTensor(feat_X_bi)
    if torch.cuda.is_available():
        features = features.to(device)
        features_bi = features_bi.to(device)
        sadj = sadj.to(device)
        num_sadj = num_sadj.to(device)


    kmeans = KMeans(n_clusters=n_clusters)
    BCE_loss = nn.BCEWithLogitsLoss()
    alpha_values = [25]
    beta_values = [0.05]
    gamma_values = [1]
    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                print(alpha, beta, gamma)
                fix_seed(2025)
                HBC_rec = []
                model = SpaMCI(nfeat=features.shape[1],
                               nhid1=config.nhid1,
                               nhid2=config.nhid2,
                               dropout=config.dropout)
                model.to(device)

                optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                model.train()
                for epoch in range(300):
                    model.train()
                    emb, emb_bi, de_emb, pi, disp, mean = model(features, features_bi, sadj)

                    emb1 = emb - torch.mean(emb, dim=0, keepdim=True)
                    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
                    emb_adj = torch.mm(emb1, emb1.T)
                    loss_adj = BCE_loss(emb_adj, num_sadj)

                    recon_loss = F.mse_loss(features, de_emb)
                    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features_bi, mean, mean=True)

                    total_loss = alpha * recon_loss + beta * zinb_loss + gamma * loss_adj

                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                model.eval()
                with torch.no_grad():
                    emb, emb_bi, de_emb, pi, disp, mean = model(features, features_bi, sadj)
                    emb = emb.cpu().numpy()
                    idx_1 = kmeans.fit(emb).labels_
                    adata.obs['domain'] = idx_1
                    adata.obs['domain'] = adata.obs['domain'].astype('category')
                    new_type = refine_label(adata, radius=50, key='domain')
                    ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], new_type)
                    print('ARI:', ARI)
                    de_emb = de_emb.cpu().numpy()
                    gene_name = adata.var.index
                    # np.save('HBC_res/de_emb.npy', de_emb)
                    # np.save('HBC_res/new_type.npy', idx_1)
                    # np.save('HBC_res/gene_name.npy', gene_name)
