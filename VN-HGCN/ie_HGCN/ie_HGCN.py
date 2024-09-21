import torch
import copy
import Data4iehgcn
import numpy as np
from torch import nn, optim, sparse
from sklearn.metrics import f1_score
from torch.nn import functional as f
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from pyg2dgl4iehgcn import data_transform
from torch_geometric.utils import dropout_edge
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

class ie_HGCN(nn.Module):
    @staticmethod
    @torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight, gain=1.414)

    def __init__(self, data, num_layers, **kwargs):
        super(ie_HGCN, self).__init__()
        self.data = data
        self.num_layers = num_layers

        # transform target type and its adjacent types to same semantic space
        self.self_transform = nn.ModuleList()
        self.semantic_layers = nn.ModuleList()

        for i in range(num_layers):
            self.self_transform.append(nn.ModuleDict())
            self.semantic_layers.append(nn.ModuleDict())
            for j in self.data.types:
                if i == 0:
                    self.self_transform[i].update(
                        {j: nn.Sequential(nn.Dropout(p=kwargs['dropout_rate']),
                                          nn.Linear(self.data.types_dim[j], kwargs['semantic_dim'][i], bias=False))})
                else:
                    self.self_transform[i].update(
                        {j: nn.Sequential(nn.Dropout(p=kwargs['dropout_rate']),
                                          nn.Linear(kwargs['semantic_dim'][i - 1], kwargs['semantic_dim'][i],
                                                    bias=False))})
                self.semantic_layers[i].update({j: nn.ModuleDict()})
                for k in self.data.adj_types[j]:
                    if i == 0:
                        self.semantic_layers[i][j].update(
                            {k: nn.Sequential(nn.Dropout(p=kwargs['dropout_rate']),
                                              nn.Linear(self.data.types_dim[k], kwargs['semantic_dim'][i],
                                                        bias=False))})
                    else:
                        self.semantic_layers[i][j].update(
                            {k: nn.Sequential(nn.Dropout(p=kwargs['dropout_rate']),
                                              nn.Linear(kwargs['semantic_dim'][i - 1], kwargs['semantic_dim'][i],
                                                        bias=False))})

        # attention
        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.values = nn.ModuleList()

        for i in range(self.num_layers):
            self.query.append(nn.ModuleDict())
            self.key.append(nn.ModuleDict())
            self.values.append(nn.ModuleDict())
            for j in self.data.types:
                self.query[i].update({j: nn.Sequential(nn.Dropout(p=kwargs['dropout_rate']),
                                                       nn.Linear(kwargs['semantic_dim'][i],
                                                                 kwargs['att_dim'], bias=False))})
                self.key[i].update({j: nn.Sequential(nn.Dropout(p=kwargs['dropout_rate']),
                                                     nn.Linear(kwargs['semantic_dim'][i],
                                                               kwargs['att_dim'], bias=False))})
                self.values[i].update({j: nn.Sequential(nn.Dropout(p=kwargs['dropout_rate']),
                                                        nn.Linear(2 * kwargs['att_dim'],
                                                                  1, bias=False), nn.ELU())})
        # classify
        self.classify_layer = nn.Sequential(
            nn.Linear(kwargs['semantic_dim'][-1], self.data.num_classes, bias=False),
            nn.Softmax(dim=1))
        self.apply(self.init_weights)

    def forward(self, training=True, drop_edge=.0):
        h = copy.deepcopy(self.data.x)
        att = [dict.fromkeys(self.data.types)]

        for i in range(self.num_layers):
            temp_h = dict.fromkeys(self.data.types)
            for j in self.data.types:
                z_self = self.self_transform[i][j](h[j])
                z_adj = dict.fromkeys(self.data.adj_types[j])
                for k in self.data.adj_types[j]:
                    drop_edges = dropout_edge(self.data.adj_m[j][k].indices(), p=drop_edge, training=training)
                    adj = torch.sparse_coo_tensor(indices=drop_edges[0],
                                                  values=self.data.adj_m[j][k].values()[drop_edges[1]],
                                                  size=(self.data.types_num[j], self.data.types_num[k]))
                    z_adj[k] = sparse.mm(adj, self.semantic_layers[i][j][k](h[k]))
                # calculate attention
                q_self = self.query[i][j](z_self)
                k_self = self.key[i][j](z_self)
                k_adj = dict.fromkeys(self.data.adj_types[j])
                # self un-normalized attention
                e = self.values[i][j](torch.concat([k_self, q_self], dim=1))
                for k in self.data.adj_types[j]:
                    k_adj[k] = self.key[i][j](z_adj[k])
                    e_adj = self.values[i][j](torch.concat([k_adj[k], q_self], dim=1))
                    e = torch.concat([e, e_adj], dim=1)
                att[i][j] = f.softmax(e, dim=1)
                temp_h[j] = torch.mul(att[i][j][:, 0].reshape(-1, 1), z_self)
                for col, k in enumerate(self.data.adj_types[j]):
                    temp_h[j] += torch.mul(att[i][j][:, col + 1].reshape(-1, 1), z_adj[k])
                temp_h[j] = f.elu(temp_h[j])
            h = temp_h
            att.append(dict.fromkeys(self.data.types))
        # embedding
        emb = h[self.data.target_type]
        # classify
        predict = self.classify_layer(h[self.data.target_type])
        return [emb, predict]


def run(n=1, dataset='dblp4MAGNN', num_layers=4, virtual_node=False, vn_num=1, **kwargs):
    ari_rec = np.zeros(n)
    nmi_rec = np.zeros(n)
    micro_f1_rec = np.zeros(n)
    macro_f1_rec = np.zeros(n)
    data = Data4iehgcn.LoadData(dataset=dataset, adj_normalized=True, virtual_node=virtual_node, vn_num=vn_num,
                                perturbation=kwargs['perturbation'],
                                p_id=kwargs['p_id'], p_std=kwargs['p_std'], p_hop=kwargs['p_hop'])
    for i in range(n):
        torch.cuda.empty_cache()
        model = ie_HGCN(data=data, num_layers=num_layers, semantic_dim=kwargs['semantic_dim'],
                        dropout_rate=kwargs['dropout'], att_dim=kwargs['att_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'])
        loss_fn = nn.CrossEntropyLoss()
        epochs = 2000
        max_valid_loss = np.inf
        early_stop = 100
        for epoch in range(epochs):
            if early_stop == 0:
                print("\nearly stop in epochs {}".format(epoch))
                break
            model.train()
            emb, pred = model(training=True, drop_edge=kwargs['drop_edge'])
            loss = loss_fn(pred[data.train_mask], data.train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                valid_loss = loss_fn(pred[data.val_mask], data.val_y)
                # early stop
                # if valid_loss < max_valid_loss:
                #     early_stop = 100
                #     max_valid_loss = valid_loss
                # else:
                #     # if virtual_node==False:
                #     #     while epoch>=300:
                #     #         early_stop -= 1
                #     # else:
                #     early_stop -= 1
                # # print validation
                if (epoch + 1) % 100 == 0:
                    print(
                        "[{} / {}]\ntrain loss: {:.4f}\nvalid loss: {:.4f}".format(epoch + 1, epochs, loss, valid_loss))
                    valid_pred = pred[data.val_mask].max(dim=1)[1]
                    micro_f1 = f1_score(data.val_y.cpu(), valid_pred.cpu(), average='micro')
                    macro_f1 = f1_score(data.val_y.cpu(), valid_pred.cpu(), average='macro')
                    print("micro_f1: {:.4f}".format(micro_f1))
                    print("macro_f1: {:.4f}".format(macro_f1))
        # test
        model.eval()
        with torch.no_grad():
            print("train loss: {:.4f}".format(loss))
            emb, pred = model(training=False, drop_edge=kwargs['drop_edge'])
            
            # visualization
            tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(emb[data.test_mask].cpu().numpy())
            plt.figure(figsize=[8.0901, 5], layout='constrained')
            plt.scatter(tsne[:, 0], tsne[:, 1], c=data.test_y.cpu().numpy())
            plt.axis('off')
            plt.savefig("ie-HGCN_{}t-sne.pdf".format(dataset))

            test_loss = loss_fn(pred[data.test_mask], data.test_y)
            test_pred = pred[data.test_mask].max(dim=1)[1]
            
            # clustering
            test_cluster = KMeans(n_clusters=data.num_classes, n_init=10).fit_predict(
                emb[data.test_mask].cpu().numpy())
            aris = ari(data.test_y.cpu().numpy(), test_cluster)
            nmis = nmi(data.test_y.cpu().numpy(), test_cluster)
            
            if kwargs['perturbation']:
                l2norm = torch.linalg.vector_norm(emb[kwargs['p_id'], :], 2).cpu().numpy()
                if virtual_node:
                    np.save('./perturbation/l2norm_vn_std{}_{}hop.npy'.format(kwargs['p_std'], kwargs['p_hop']), l2norm)
                else:
                    np.save('./perturbation/l2norm_std{}_{}hop.npy'.format(kwargs['p_std'], kwargs['p_hop']), l2norm)
                print("l2 norm of embedding of perturbed node:{:.4f}".format(l2norm))
            else:
                l2norm = torch.linalg.vector_norm(emb[kwargs['p_id'], :], 2).cpu().numpy()
                if virtual_node:
                    np.save('./perturbation/l2norm_vn_{}hop.npy'.format(kwargs['p_hop']), l2norm)
                else:
                    np.save('./perturbation/l2norm_{}hop.npy'.format(kwargs['p_hop']), l2norm)
                print("l2 norm of embedding of unperturbed node:{:.4f}".format(l2norm))
            
            micro_f1 = f1_score(data.test_y.cpu(), test_pred.cpu(), average='micro')
            macro_f1 = f1_score(data.test_y.cpu(), test_pred.cpu(), average='macro')
            print("\ntest:")
            print("Loss :{:.4f}".format(test_loss))
            print("test ARI :{:.4f}".format(aris))
            print("test NMI :{:.4f}".format(nmis))
            print("micro_f1: {:.4f}".format(micro_f1))
            print("macro_f1: {:.4f}".format(macro_f1))
            ari_rec[i] = aris
            nmi_rec[i] = nmis
            micro_f1_rec[i] = micro_f1
            macro_f1_rec[i] = macro_f1

    print("\n" + "*" * 10)
    print("ARI: {:.4f} +/- {:.4f}".format(ari_rec.mean(), ari_rec.std()))
    print("NMI: {:.4f} +/- {:.4f}".format(nmi_rec.mean(), nmi_rec.std()))
    print("Micro_f1: {:.4f} +/- {:.4f}".format(micro_f1_rec.mean(), micro_f1_rec.std()))
    print("Macro_f1: {:.4f} +/- {:.4f}".format(macro_f1_rec.mean(), macro_f1_rec.std()))
    
    return micro_f1_rec.mean()