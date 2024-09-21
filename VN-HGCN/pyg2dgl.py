import torch
from torch_geometric import datasets
import numpy as np
from sklearn.model_selection import train_test_split as tts
import dgl
import openhgnn
import dgl.sparse as dglsp


def data_transform(dataset='dblp4MAGNN', train_size=.5, random_state=0):
    if dataset == 'dblp4MAGNN':
        data = datasets.DBLP("./dataset/DBLP").data
        A_P = data['author', 'paper'].edge_index.reshape(-1, ).chunk(2)
        T_P = data['term', 'paper'].edge_index.reshape(-1, ).chunk(2)
        V_P = data['conference', 'paper'].edge_index.reshape(-1, ).chunk(2)
        P_A = data['paper', 'author'].edge_index.reshape(-1, ).chunk(2)
        P_T = data['paper', 'term'].edge_index.reshape(-1, ).chunk(2)
        P_V = data['paper', 'conference'].edge_index.reshape(-1, ).chunk(2)

        g = dgl.heterograph({
            ('A', 'A-P', 'P'): A_P,
            ('T', 'T-P', 'P'): T_P,
            ('V', 'V-P', 'P'): V_P,
            ('P', 'P-A', 'A'): P_A,
            ('P', 'P-T', 'T'): P_T,
            ('P', 'P-V', 'V'): P_V,
        })

        g.ndata['h'] = {'A': data['author']['x'], 'P': data['paper']['x'],
                        'T': data['term']['x'], 'V': torch.eye(data['conference'].num_nodes)}
        
        g.ndata['labels'] = {'A': data['author']['y']}

        ind = np.arange(data['author'].num_nodes)
        train_ind, test_ind = tts(ind, train_size=train_size, random_state=random_state)
        test_ind, valid_ind = tts(test_ind, train_size=.5, random_state=random_state)

        mask = np.zeros(data['author'].num_nodes)
        mask[train_ind] = 1
        g.ndata['train_mask'] = {'A': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['author'].num_nodes)
        mask[valid_ind] = 1
        g.ndata['val_mask'] = {'A': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['author'].num_nodes)
        mask[test_ind] = 1
        g.ndata['test_mask'] = {'A': torch.tensor(mask, dtype=torch.int8)}

    elif dataset == 'imdb4MAGNN':
        data = datasets.IMDB("./dataset/IMDB").data
        A_M = data['actor', 'movie'].edge_index.reshape(-1, ).chunk(2)
        D_M = data['director', 'movie'].edge_index.reshape(-1, ).chunk(2)
        M_D = data['movie', 'director'].edge_index.reshape(-1, ).chunk(2)
        M_A = data['movie', 'actor'].edge_index.reshape(-1, ).chunk(2)

        g = dgl.heterograph({
            ('A', 'A-M', 'M'): A_M,
            ('D', 'D-M', 'M'): D_M,
            ('M', 'M-A', 'A'): M_A,
            ('M', 'M-D', 'D'): M_D,
        })

        g.ndata['h'] = {'A': data['actor']['x'], 'D': data['director']['x'], 'M': data['movie']['x'], }
        g.ndata['labels'] = {'M': data['movie']['y']}

        ind = np.arange(data['movie'].num_nodes)
        train_ind, test_ind = tts(ind, train_size=train_size, random_state=random_state)
        test_ind, valid_ind = tts(test_ind, train_size=.5, random_state=random_state)

        mask = np.zeros(data['movie'].num_nodes)
        mask[train_ind] = 1
        g.ndata['train_mask'] = {'M': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['movie'].num_nodes)
        mask[valid_ind] = 1
        g.ndata['val_mask'] = {'M': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['movie'].num_nodes)
        mask[test_ind] = 1
        g.ndata['test_mask'] = {'M': torch.tensor(mask, dtype=torch.int8)}
    
    elif dataset == 'acm4NSHE':
        g = openhgnn.dataset.AcademicDataset(name='{}'.format(dataset))[0]
        
        ind = np.arange(g.num_nodes('paper'))
        train_ind, test_ind = tts(ind, train_size=.5)
        test_ind, valid_ind = tts(test_ind, train_size=.5)

        mask = np.zeros(g.num_nodes('paper'))
        mask[train_ind] = 1
        g.ndata['train_mask'] = {'paper': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(g.num_nodes('paper'))
        mask[valid_ind] = 1
        g.ndata['val_mask'] = {'paper': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(g.num_nodes('paper'))
        mask[test_ind] = 1
        g.ndata['test_mask'] = {'paper': torch.tensor(mask, dtype=torch.int8)}
    
    elif dataset == 'ogbn-mag':
        data = datasets.OGB_MAG("./dataset/OGB_MAG").data
        
        A_I = data['author', 'institution'].edge_index.reshape(-1, ).chunk(2)
        A_P = data['author', 'paper'].edge_index.reshape(-1, ).chunk(2)
        P_F = data['paper', 'field_of_study'].edge_index.reshape(-1, ).chunk(2)
        P_P = data['paper', 'paper'].edge_index.reshape(-1, ).chunk(2)
        
        # I_A = data['author', 'institution'].edge_index[[1,0]].reshape(-1, ).chunk(2)
        # P_A = data['author', 'paper'].edge_index[[1,0]].reshape(-1, ).chunk(2)
        # F_P = data['paper', 'field_of_study'].edge_index[[1,0]].reshape(-1, ).chunk(2)
        # P_P1 = data['paper', 'paper'].edge_index[[1,0]].reshape(-1, ).chunk(2)

        g = dgl.heterograph({
            ('A', 'A-I', 'I'): A_I,
            ('A', 'A-P', 'P'): A_P,
            ('P', 'P-F', 'F'): P_F,
            ('P', 'P-P0', 'P'): P_P,
            # ('I', 'I-A', 'A'): I_A,
            # ('P', 'P-A', 'A'): P_A,
            # ('F', 'F-P', 'P'): F_P,
            # ('P', 'P-P1', 'P'): P_P1
        })

        # g.ndata['h'] = {'A': torch.eye(data['author'].num_nodes), 'P': data['paper']['x'],
        #                 'I': torch.eye(data['institution'].num_nodes), 'F': torch.eye(data['field_of_study'].num_nodes)}
        
        g.ndata['h'] = {'A': sparse_coo_eye(data['author'].num_nodes), 'P': data['paper']['x'],
                        'I': sparse_coo_eye(data['institution'].num_nodes), 'F': sparse_coo_eye(data['field_of_study'].num_nodes)}
        
        g.ndata['labels'] = {'P': data['paper']['y']}

        ind = np.arange(data['paper'].num_nodes)
        train_ind, test_ind = tts(ind, train_size=train_size, random_state=random_state)
        test_ind, valid_ind = tts(test_ind, train_size=.5, random_state=random_state)

        mask = np.zeros(data['paper'].num_nodes)
        mask[train_ind] = 1
        g.ndata['train_mask'] = {'P': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['paper'].num_nodes)
        mask[valid_ind] = 1
        g.ndata['val_mask'] = {'P': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['paper'].num_nodes)
        mask[test_ind] = 1
        g.ndata['test_mask'] = {'P': torch.tensor(mask, dtype=torch.int8)}
        
    elif dataset == 'aminer':
        data = datasets.AMiner("./dataset/AMiner").data
        
        A_I = data['author', 'institution'].edge_index.reshape(-1, ).chunk(2)
        A_P = data['author', 'paper'].edge_index.reshape(-1, ).chunk(2)
        P_F = data['paper', 'field_of_study'].edge_index.reshape(-1, ).chunk(2)
        P_P = data['paper', 'paper'].edge_index.reshape(-1, ).chunk(2)
        
        # I_A = data['author', 'institution'].edge_index[[1,0]].reshape(-1, ).chunk(2)
        # P_A = data['author', 'paper'].edge_index[[1,0]].reshape(-1, ).chunk(2)
        # F_P = data['paper', 'field_of_study'].edge_index[[1,0]].reshape(-1, ).chunk(2)
        # P_P1 = data['paper', 'paper'].edge_index[[1,0]].reshape(-1, ).chunk(2)

        g = dgl.heterograph({
            ('A', 'A-I', 'I'): A_I,
            ('A', 'A-P', 'P'): A_P,
            ('P', 'P-F', 'F'): P_F,
            ('P', 'P-P0', 'P'): P_P,
            # ('I', 'I-A', 'A'): I_A,
            # ('P', 'P-A', 'A'): P_A,
            # ('F', 'F-P', 'P'): F_P,
            # ('P', 'P-P1', 'P'): P_P1
        })

        # g.ndata['h'] = {'A': torch.eye(data['author'].num_nodes), 'P': data['paper']['x'],
        #                 'I': torch.eye(data['institution'].num_nodes), 'F': torch.eye(data['field_of_study'].num_nodes)}
        
        g.ndata['h'] = {'A': sparse_coo_eye(data['author'].num_nodes), 'P': data['paper']['x'],
                        'I': sparse_coo_eye(data['institution'].num_nodes), 'F': sparse_coo_eye(data['field_of_study'].num_nodes)}
        
        g.ndata['labels'] = {'P': data['paper']['y']}

        ind = np.arange(data['paper'].num_nodes)
        train_ind, test_ind = tts(ind, train_size=train_size, random_state=random_state)
        test_ind, valid_ind = tts(test_ind, train_size=.5, random_state=random_state)

        mask = np.zeros(data['paper'].num_nodes)
        mask[train_ind] = 1
        g.ndata['train_mask'] = {'P': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['paper'].num_nodes)
        mask[valid_ind] = 1
        g.ndata['val_mask'] = {'P': torch.tensor(mask, dtype=torch.int8)}

        mask = np.zeros(data['paper'].num_nodes)
        mask[test_ind] = 1
        g.ndata['test_mask'] = {'P': torch.tensor(mask, dtype=torch.int8)}
        
    else:
        raise Exception("check the name of dataset, make sure input right dataset")
    dgl.save_graphs("./dataset/{}.bin".format(dataset), g)

    
def sparse_coo_eye(size):
    ind = torch.arange(size).repeat(2,1)
    v = torch.ones(size)
    res = torch.sparse_coo_tensor(ind,v,[size,size])
    print(res)
    return res

# def sparse_coo_eye(size):
#     ind = torch.arange(size).repeat(2,1)
#     res = dglsp.spmatrix(ind)
#     res = dglsp.identity([size,size])
#     return res

# def sparse_coo_eye(size):
#     crow = torch.arange(size+1)
#     col_ind = torch.arange(size)
#     v = torch.ones(size)
    
#     res = torch.sparse_csr_tensor(crow, col_ind, v)
#     return res

# def sparse_coo_eye(size):
    # ind = torch.arange(size).repeat(2,1)
    # res = dglsp.from_coo(ind[0],ind[1])
    # return res