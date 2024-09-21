import torch
import copy
import Data
import numpy as np
from torch import nn, optim, sparse
from torch_geometric.utils import dropout_edge
from sklearn.metrics import f1_score
from torch.nn import functional as f
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, data, **kwargs):
        super(Model, self).__init__()
        self.data = data
        self.num_layers = kwargs['num_layers']
        # transform target type and its adjacent types to same semantic space
        self.self_transform = nn.ModuleList()
        self.semantic_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.self_transform.append(nn.ModuleDict())
            self.semantic_layers.append(nn.ModuleDict())
            for j in self.data.types:
                if i == 0:
                    self.self_transform[i].update(
                        {j: nn.Sequential(nn.Dropout(p=kwargs['dropout']),
                                          nn.Linear(self.data.types_dim[j], kwargs['hidden_dim'], bias=False))})
                else:
                    self.self_transform[i].update(
                        {j: nn.Sequential(nn.Dropout(p=kwargs['dropout']),
                                          nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim'], bias=False))})
                self.semantic_layers[i].update({j: nn.ModuleDict()})
                for k in self.data.adj_types[j]:
                    if i == 0:
                        self.semantic_layers[i][j].update(
                            {k: nn.Sequential(nn.Dropout(p=kwargs['dropout']),
                                              nn.Linear(self.data.types_dim[k], kwargs['hidden_dim'], bias=False))})
                    else:
                        self.semantic_layers[i][j].update(
                            {k: nn.Sequential(nn.Dropout(p=kwargs['dropout']),
                                              nn.Linear(kwargs['hidden_dim'], kwargs['hidden_dim'], bias=False))})

        # attention
        self.attention = nn.ModuleList()
        for i in range(self.num_layers):
            self.attention.append(nn.ModuleDict())
            for j in self.data.types:
                self.attention[i].update({j: nn.Sequential(nn.Dropout(p=kwargs['dropout']),
                                                           nn.Linear(kwargs['hidden_dim'], kwargs['att_dim'],
                                                                     bias=False),
                                                           nn.Linear(kwargs['att_dim'], 1, bias=False),
                                                           nn.Tanh())})

        # classify
        self.classify_layer = nn.Sequential(nn.Linear(kwargs['hidden_dim'], self.data.num_classes, bias=False),
                                            nn.Softmax(dim=1))

    def forward(self, training=True, drop_edge=.5):
        h = copy.deepcopy(self.data.x)
        for i in range(self.num_layers):
            temp_h = dict.fromkeys(self.data.types)
            for j in self.data.types:
                z_self = self.self_transform[i][j](h[j])
                att = self.attention[i][j](z_self)
                z_adj = dict.fromkeys(self.data.adj_types[j])
                for k in self.data.adj_types[j]:
                    drop_edges = dropout_edge(self.data.adj_m[j][k].indices(), p=drop_edge, training=training)
                    adj = torch.sparse_coo_tensor(indices=drop_edges[0],
                                                  values=self.data.adj_m[j][k].values()[drop_edges[1]],
                                                  size=(self.data.types_num[j], self.data.types_num[k]))
                    z_adj[k] = sparse.mm(adj, self.semantic_layers[i][j][k](h[k]))
                    att = torch.concat([att, self.attention[i][j](z_adj[k])])
                att = f.softmax(att, dim=1)
                temp_h[j] = torch.mul(att[0], z_self)
                for col, k in enumerate(self.data.adj_types[j]):
                    temp_h[j] += torch.mul(att[col + 1], z_adj[k])
                temp_h[j] = f.relu(temp_h[j])
            h = temp_h

        # embedding
        emb = h[self.data.target_type]
        # classify
        predict = self.classify_layer(emb)
        return emb, predict

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight, gain=nn.init.calculate_gain('relu'))
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))


def run(n=1, dataset='dblp4MAGNN', num_layers=4, initialize=False, **kwargs):
    ari_rec = np.zeros(n)
    nmi_rec = np.zeros(n)
    micro_f1_rec = np.zeros(n)
    macro_f1_rec = np.zeros(n)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 1000
    data = Data.LoadData(dataset=dataset, adj_normalized=True)
    for i in range(n):
        torch.cuda.empty_cache()
        model = Model(data=data, num_layers=num_layers, hidden_dim=kwargs['hidden_dim'],
                      dropout=kwargs['dropout'], att_dim=kwargs['att_dim']).to(device)
        if initialize:
            model.initialize()
        optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'], weight_decay=kwargs['weight_decay'])
        max_valid_loss = np.inf
        early_stop = 100
        for epoch in range(epochs):
            if early_stop == 0:
                print("\nearly stop in epoch {}".format(epoch))
                break
            model.train()
            optimizer.zero_grad()
            emb, pred = model(training=True, drop_edge=kwargs['drop_edge'])
            loss = loss_fn(pred[data.train_mask], data.train_y)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                emb, pred = model(training=False)
                valid_loss = loss_fn(pred[data.val_mask], data.val_y)
                # early stop
                if valid_loss <= max_valid_loss:
                    early_stop = 100
                    max_valid_loss = valid_loss
                else:
                    if epoch >= 300:
                        early_stop -= 1
        # test
        model.eval()
        with torch.no_grad():
            emb, pred = model(training=False)
            # visualization
            tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(emb[data.test_mask].cpu().numpy())
            plt.figure(figsize=[8.0901, 5], layout='constrained')
            plt.scatter(tsne[:, 0], tsne[:, 1], c=data.test_y.cpu().numpy())
            plt.axis('off')
            plt.savefig("./output/VN-HGCN_{}_t-sne.pdf".format(dataset))

            test_loss = loss_fn(pred[data.test_mask], data.test_y)
            test_pred = pred[data.test_mask].max(dim=1)[1]
            
            test_cluster = KMeans(n_clusters=data.num_classes, n_init=10).fit_predict(
                emb[data.test_mask].cpu().numpy())
            aris = ari(data.test_y.cpu().numpy(), test_cluster)
            nmis = nmi(data.test_y.cpu().numpy(), test_cluster)
            
            micro_f1 = f1_score(data.test_y.cpu(), test_pred.cpu(), average='micro')
            macro_f1 = f1_score(data.test_y.cpu(), test_pred.cpu(), average='macro')
            print("[{}/{}] result:".format(i + 1, n))
            print("train loss :{:.4f}".format(loss))
            print("test loss :{:.4f}".format(test_loss))
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
    np.save("./output/{}VN-HGCN_ari".format(dataset), ari_rec)
    np.save("./output/{}VN-HGCN_nmi".format(dataset), nmi_rec)
    np.save("./output/{}VN-HGCN_micro_f1".format(dataset), micro_f1_rec)
    np.save("./output/{}VN-HGCN_macro_f1".format(dataset), macro_f1_rec)
    print("\n")
