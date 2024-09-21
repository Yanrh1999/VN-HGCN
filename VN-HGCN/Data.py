import dgl
import torch
import numpy as np


class LoadData:
    def __init__(self, dataset='dblp4MAGNN', adj_normalized=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = dgl.load_graphs("./dataset/{}_vn.bin".format(dataset))[0][0]
        if dataset == "dblp4MAGNN":
            self.target_type = 'A'
        elif dataset == "imdb4MAGNN":
            self.target_type = 'M'
        elif dataset == "acm4NSHE":
            self.target_type = 'paper'
            g.ndata['labels'] = g.ndata['label']
        elif dataset == 'ogbn-mag':
            self.target_type = 'paper'
            g.ndata['labels'] = g.ndata['labels']
        self.types = g.ntypes
        self.types_dim = {}
        self.types_dim.update({i: g.ndata['h'][i].shape[-1] for i in g.ntypes})
        self.types_num = dict(zip(g.ntypes, [g.num_nodes(i) for i in g.ntypes]))
        self.num_classes = len(np.unique(g.ndata['labels'][self.target_type]))

        self.train_mask = g.ndata['train_mask'][self.target_type].to(torch.bool)
        self.val_mask = g.ndata['val_mask'][self.target_type].to(torch.bool)
        self.test_mask = g.ndata['test_mask'][self.target_type].to(torch.bool)

        self.train_y = g.ndata['labels'][self.target_type][self.train_mask].to(device)
        self.val_y = g.ndata['labels'][self.target_type][self.val_mask].to(device)
        self.test_y = g.ndata['labels'][self.target_type][self.test_mask].to(device)

        self.x = dict(zip(g.ntypes, [g.ndata['h'][i].to(device) for i in g.ntypes]))
        
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
