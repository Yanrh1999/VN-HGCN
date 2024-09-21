import dgl
import torch

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)

def add_vn(g: dgl.DGLGraph, vn_num=1, vn_dim=64, dataset='dblp4MAGNN', vn_initialize='mean', central_vn_num=1):
    # dblp4MAGNN
    if dataset == 'dblp4MAGNN':
        A_P = g.edges(etype='A-P')
        P_A = g.edges(etype='P-A')
        T_P = g.edges(etype='T-P')
        P_T = g.edges(etype='P-T')
        V_P = g.edges(etype='V-P')
        P_V = g.edges(etype='P-V')

        A_SA, SA_A = vn_edge(g.num_nodes('A'), vn_num)
        P_SP, SP_P = vn_edge(g.num_nodes('P'), vn_num)
        T_ST, ST_T = vn_edge(g.num_nodes('T'), vn_num)
        V_SV, SV_V = vn_edge(g.num_nodes('V'), vn_num)

        SA_S, S_SA = vn_edge(vn_num, central_vn_num, random=True)
        SP_S, S_SP = vn_edge(vn_num, central_vn_num, random=True)
        ST_S, S_ST = vn_edge(vn_num, central_vn_num, random=True)
        SV_S, S_SV = vn_edge(vn_num, central_vn_num, random=True)

        new_g = dgl.heterograph({
            ('A', 'A-P', 'P'): A_P,
            ('T', 'T-P', 'P'): T_P,
            ('V', 'V-P', 'P'): V_P,
            ('P', 'P-A', 'A'): P_A,
            ('P', 'P-T', 'T'): P_T,
            ('P', 'P-V', 'V'): P_V,
            ('A', 'A-SA', 'SA'): A_SA,
            ('P', 'P-SP', 'SP'): P_SP,
            ('T', 'T-ST', 'ST'): T_ST,
            ('V', 'V-SV', 'SV'): V_SV,
            ('SA', 'SA-A', 'A'): SA_A,
            ('SP', 'SP-P', 'P'): SP_P,
            ('ST', 'ST-T', 'T'): ST_T,
            ('SV', 'SV-V', 'V'): SV_V,
            ('SA', 'SA-S', 'S'): SA_S,
            ('SP', 'SP-S', 'S'): SP_S,
            ('ST', 'ST-S', 'S'): ST_S,
            ('SV', 'SV-S', 'S'): SV_S,
            ('S', 'S-SA', 'SA'): S_SA,
            ('S', 'S-SP', 'SP'): S_SP,
            ('S', 'S-ST', 'ST'): S_ST,
            ('S', 'S-SV', 'SV'): S_SV
        })
        if vn_initialize == 'ones':
            new_g.ndata['h'] = {'A': g.ndata['h']['A'], 'P': g.ndata['h']['P'],
                                'T': g.ndata['h']['T'], 'V': g.ndata['h']['V'],
                                'SA': torch.ones([vn_num, vn_dim]),
                                'SP': torch.ones([vn_num, vn_dim]),
                                'ST': torch.ones([vn_num, vn_dim]),
                                'SV': torch.ones([vn_num, vn_dim]),
                                'S': torch.ones([central_vn_num, vn_dim])}
        elif vn_initialize == 'mean':
            SA = torch.vstack(
                [torch.mean(g.ndata['h']['A'][SA_A[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            SP = torch.vstack(
                [torch.mean(g.ndata['h']['P'][SP_P[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            ST = torch.vstack(
                [torch.mean(g.ndata['h']['T'][ST_T[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            SV = torch.vstack(
                [torch.mean(g.ndata['h']['V'][SV_V[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            new_g.ndata['h'] = {'A': g.ndata['h']['A'], 'P': g.ndata['h']['P'],
                                'T': g.ndata['h']['T'], 'V': g.ndata['h']['V'],
                                'SA': SA,
                                'SP': SP,
                                'ST': ST,
                                'SV': SV,
                                'S': torch.ones([central_vn_num, vn_dim])}
        new_g.ndata['labels'] = g.ndata['labels']

    # imdb4MAGNN
    elif dataset == "imdb4MAGNN":
        A_M = g.edges(etype='A-M')
        D_M = g.edges(etype='D-M')
        M_A = g.edges(etype='M-A')
        M_D = g.edges(etype='M-D')

        A_SA, SA_A = vn_edge(g.num_nodes('A'), vn_num)
        D_SD, SD_D = vn_edge(g.num_nodes('D'), vn_num)
        M_SM, SM_M = vn_edge(g.num_nodes('M'), vn_num)

        SA_S, S_SA = vn_edge(vn_num, central_vn_num, random=True)
        SD_S, S_SD = vn_edge(vn_num, central_vn_num, random=True)
        SM_S, S_SM = vn_edge(vn_num, central_vn_num, random=True)

        '''
        S_SA = (torch.tensor([0] * vn_num), torch.arange(vn_num))
        S_SD = (torch.tensor([0] * vn_num), torch.arange(vn_num))
        S_SM = (torch.tensor([0] * vn_num), torch.arange(vn_num))
        SA_S = (torch.arange(vn_num), torch.tensor([0] * vn_num))
        SD_S = (torch.arange(vn_num), torch.tensor([0] * vn_num))
        SM_S = (torch.arange(vn_num), torch.tensor([0] * vn_num))
        '''

        new_g = dgl.heterograph({
            ('A', 'A-M', 'M'): A_M,
            ('D', 'D-M', 'M'): D_M,
            ('M', 'M-A', 'A'): M_A,
            ('M', 'M-D', 'D'): M_D,
            ('A', 'A-SA', 'SA'): A_SA,
            ('D', 'D-SD', 'SD'): D_SD,
            ('M', 'M-SM', 'SM'): M_SM,
            ('SA', 'SA-A', 'A'): SA_A,
            ('SD', 'SD-D', 'D'): SD_D,
            ('SM', 'SM-M', 'M'): SM_M,
            ('S', 'S-SA', 'SA'): S_SA,
            ('S', 'S-SD', 'SD'): S_SD,
            ('S', 'S-SM', 'SM'): S_SM,
            ('SA', 'SA-S', 'S'): SA_S,
            ('SD', 'SD-S', 'S'): SD_S,
            ('SM', 'SM-S', 'S'): SM_S,
        })
        if vn_initialize == 'ones':
            new_g.ndata['h'] = {'A': g.ndata['h']['A'], 'D': g.ndata['h']['D'], 'M': g.ndata['h']['M'],
                                'SA': torch.ones([vn_num, vn_dim]),
                                'SD': torch.ones([vn_num, vn_dim]),
                                'SM': torch.ones([vn_num, vn_dim]),
                                'S': torch.ones([central_vn_num, vn_dim])}
        elif vn_initialize == 'mean':
            SA = torch.vstack(
                [torch.mean(g.ndata['h']['A'][SA_A[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            SD = torch.vstack(
                [torch.mean(g.ndata['h']['D'][SD_D[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            SM = torch.vstack(
                [torch.mean(g.ndata['h']['M'][SM_M[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            new_g.ndata['h'] = {'A': g.ndata['h']['A'], 'D': g.ndata['h']['D'], 'M': g.ndata['h']['M'],
                                'SA': SA,
                                'SD': SD,
                                'SM': SM,
                                'S': torch.ones([central_vn_num, vn_dim])}
        new_g.ndata['labels'] = g.ndata['labels']

    # acm4NSHE
    elif dataset == 'acm4NSHE':
        A_P = g.edges(etype='author-paper')
        P_A = g.edges(etype='paper-author')
        P_S = g.edges(etype='paper-subject')
        S_P = g.edges(etype='subject-paper')

        A_SA, SA_A = vn_edge(g.num_nodes('author'), vn_num)
        P_SP, SP_P = vn_edge(g.num_nodes('paper'), vn_num)
        S_SS, SS_S = vn_edge(g.num_nodes('subject'), vn_num)

        SA_C, C_SA = vn_edge(vn_num, central_vn_num, random=True)
        SP_C, C_SP = vn_edge(vn_num, central_vn_num, random=True)
        SS_C, C_SS = vn_edge(vn_num, central_vn_num, random=True)

        '''
        C_SA = (torch.tensor([0] * vn_num), torch.arange(vn_num))
        C_SP = (torch.tensor([0] * vn_num), torch.arange(vn_num))
        C_SS = (torch.tensor([0] * vn_num), torch.arange(vn_num))
        SA_C = (torch.arange(vn_num), torch.tensor([0] * vn_num))
        SP_C = (torch.arange(vn_num), torch.tensor([0] * vn_num))
        SS_C = (torch.arange(vn_num), torch.tensor([0] * vn_num))
        '''

        new_g = dgl.heterograph({
            ('author', 'author-paper', 'paper'): A_P,
            ('paper', 'paper-author', 'author'): P_A,
            ('paper', 'paper-subject', 'subject'): P_S,
            ('subject', 'subject-paper', 'paper'): S_P,
            ('author', 'author-SA', 'SA'): A_SA,
            ('paper', 'paper-SP', 'SP'): P_SP,
            ('subject', 'subject-SS', 'SS'): S_SS,
            ('SA', 'SA-author', 'author'): SA_A,
            ('SP', 'SP-paper', 'paper'): SP_P,
            ('SS', 'SS-subject', 'subject'): SS_S,
            ('SA', 'SA-C', 'C'): SA_C,
            ('SP', 'SP-C', 'C'): SP_C,
            ('SS', 'SS-C', 'C'): SS_C,
            ('C', 'C-SA', 'SA'): C_SA,
            ('C', 'C-SP', 'SP'): C_SP,
            ('C', 'C-SS', 'SS'): C_SS,
        })
        if vn_initialize == 'ones':
            new_g.ndata['h'] = {'author': g.ndata['h']['author'],
                                'paper': g.ndata['h']['paper'],
                                'subject': g.ndata['h']['subject'],
                                'SA': torch.ones([vn_num, vn_dim]),
                                'SP': torch.ones([vn_num, vn_dim]),
                                'SS': torch.ones([vn_num, vn_dim]),
                                'C': torch.ones([central_vn_num, vn_dim])}
        elif vn_initialize == 'mean':
            SA = torch.vstack(
                [torch.mean(g.ndata['h']['author'][SA_A[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            SP = torch.vstack(
                [torch.mean(g.ndata['h']['paper'][SP_P[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            SS = torch.vstack(
                [torch.mean(g.ndata['h']['subject'][SS_S[0] == i, :], dim=0).reshape(1, -1) for i in range(vn_num)])
            new_g.ndata['h'] = {'author': g.ndata['h']['author'],
                                'paper': g.ndata['h']['paper'],
                                'subject': g.ndata['h']['subject'],
                                'SA': SA,
                                'SP': SP,
                                'SS': SS,
                                'C': torch.ones([central_vn_num, vn_dim])}   
        new_g.ndata['labels'] = g.ndata['label']

    new_g.ndata['train_mask'] = g.ndata['train_mask']
    new_g.ndata['val_mask'] = g.ndata['val_mask']
    new_g.ndata['test_mask'] = g.ndata['test_mask']
    dgl.save_graphs("../dataset/{}_vn.bin".format(dataset), new_g)


def vn_edge(num_nodes, vn_num, random=True):
    if random:
        temp = torch.arange(vn_num)
        ind = temp.repeat((num_nodes//vn_num)+1)[:num_nodes][torch.randperm(num_nodes)]
        # ind = torch.randint(0, vn_num, [num_nodes, ])
    else:
        ind = torch.zeros(num_nodes, dtype=torch.int64)
        chunks = torch.chunk(torch.arange(num_nodes), vn_num)
        for i, j in enumerate(chunks):
            ind[j] = i
    return (torch.arange(num_nodes), ind), (ind, torch.arange(num_nodes))
