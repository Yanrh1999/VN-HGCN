import dgl
import torch
import numpy as np
from Add_VN import add_vn

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

class LoadData:
    def __init__(self, dataset='dblp4MAGNN', adj_normalized=True, virtual_node=False, vn_num=1,
                 perturbation=False, p_id=0, p_std=1e0, p_hop=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = dgl.load_graphs("../dataset/{}.bin".format(dataset))[0][0]
        if virtual_node:
            print("Add Virtual Node!")
            add_vn(g, vn_num=vn_num, dataset=dataset)
            g = dgl.load_graphs("../dataset/{}_vn.bin".format(dataset))[0][0]
        else:
            pass

        if dataset == "dblp4MAGNN":
            self.target_type = 'A'
        elif dataset == "imdb4MAGNN":
            self.target_type = 'M'
        elif dataset == "acm4NSHE":
            self.target_type = 'paper'

        self.x = dict(zip(g.ntypes, [g.ndata['h'][i].to(device) for i in g.ntypes]))

        # perturbation
        if perturbation:
            self.p_id = p_id
            if p_hop != 0:
                print("adding Guassian noise with std:{}".format(p_std))
                
                og = dgl.load_graphs("../dataset/{}.bin".format(dataset))[0][0]
                
                sub_g0, _= dgl.khop_in_subgraph(og, {self.target_type: p_id}, k=p_hop, relabel_nodes=True, store_ids=True)
                sub_g1, _= dgl.khop_in_subgraph(og, {self.target_type: p_id}, k=p_hop-1, relabel_nodes=True, store_ids=True)
                sub_g2, _= dgl.khop_in_subgraph(og, {self.target_type: p_id}, k=p_hop-2, relabel_nodes=True, store_ids=True)
                
                id = dict.fromkeys(og.ntypes)
                for i in og.ntypes:
                    temp_mask0 = ~torch.isin(sub_g0.ndata[dgl.NID][i], sub_g1.ndata[dgl.NID][i])
                    temp_mask1 = ~torch.isin(sub_g0.ndata[dgl.NID][i], sub_g2.ndata[dgl.NID][i])
                    id[i] = sub_g0.ndata[dgl.NID][i][temp_mask0 & temp_mask1]
                # for i in self.x.keys():
                    self.x[i][id[i], :] += torch.normal(0, p_std, [id[i].shape[0],self.x[i].shape[1]]).to(device)
            else:
                print("without perturbation")
        else:
            print("without perturbation")

        self.types = g.ntypes
        self.types_dim = {}
        self.types_dim.update({i: g.ndata['h'][i].shape[-1] for i in g.ntypes})
        self.types_num = dict(zip(g.ntypes, [g.num_nodes(i) for i in g.ntypes]))
        
        if (virtual_node==False) and (dataset=='acm4NSHE'):
            g.ndata['labels'] = g.ndata['label']
        self.num_classes = len(np.unique(g.ndata['labels'][self.target_type]))

        self.train_mask = g.ndata['train_mask'][self.target_type].to(torch.bool)
        self.val_mask = g.ndata['val_mask'][self.target_type].to(torch.bool)
        self.test_mask = g.ndata['test_mask'][self.target_type].to(torch.bool)

        print("train num:{}".format(g.ndata['train_mask'][self.target_type].count_nonzero()))
        print("val num:{}".format(g.ndata['val_mask'][self.target_type].count_nonzero()))
        print("test num:{}".format(g.ndata['test_mask'][self.target_type].count_nonzero()))

        self.train_y = g.ndata['labels'][self.target_type][self.train_mask].to(device)
        self.val_y = g.ndata['labels'][self.target_type][self.val_mask].to(device)
        self.test_y = g.ndata['labels'][self.target_type][self.test_mask].to(device)

        self.adj_types = dict.fromkeys(g.ntypes)
        for et in g.etypes:
            t0, t1 = et.split(sep='-')
            if self.adj_types[t0] is not None:
                self.adj_types[t0].append(t1)
            else:
                self.adj_types.update({t0: []})
                self.adj_types[t0].append(t1)

        self.adj_m = dict.fromkeys(self.types)
        for i in self.types:
            self.adj_m.update({i: dict.fromkeys(self.adj_types[i])})
            for j in self.adj_types[i]:
                self.adj_m[i][j] = get_adj(g, i, j, normalized=adj_normalized).to(device)
        del g


def get_adj(g, type0, type1, normalized=True):
    edges = g.edges(etype='{}-{}'.format(type0, type1))
    indices = torch.vstack([edges[0], edges[1]])
    values = torch.ones(indices.shape[-1])
    adj = torch.sparse_coo_tensor(indices, values, [g.num_nodes(type0), g.num_nodes(type1)],
                                  dtype=torch.float).coalesce()
    if normalized:
        dinv = torch.diag(1 / torch.count_nonzero(adj.to_dense(), dim=1)).to_sparse_coo()
        adj = torch.sparse.mm(dinv, adj).coalesce()
    return adj
