from ie_HGCN import run
from pyg2dgl4iehgcn import data_transform
import argparse
import warnings


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', default='dblp4MAGNN', type=str, help='name of datasets')
    parser.add_argument('--add_vn', '-v', action="store_true", help='adding virtual nodes or not')
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--layers', '-n', default=4, type=int, help='number of layers')
    parser.add_argument('--vn_num', '-vn', default=2, type=int, help='number of virtual nodes')
    parser.add_argument('--times', '-t', default=5, type=int, help='number of runs')
    parser.add_argument('--train_size', '-ts', default=.4, type=float, help='training size')
    parser.add_argument('--drop_rate', '-dr', default=.5, type=float, help='dropout rate and dropedge rate')
    parser.add_argument('--perturbation', '-p', action="store_true", help='perturbation or not')
    args = parser.parse_args()
    
    # dataset setting
    dataset = args.dataset
    train_size = args.train_size
    random_state = 0
    data_transform(dataset=dataset, train_size=train_size, random_state=random_state)
    
    # adding virtual nodes or not
    virtual_node = args.add_vn
    vn_num = args.vn_num

    run_times = args.times
    num_layers = args.layers
    semantic_dim = [64,64,64,64]
    lr = args.learning_rate
    weight_decay = 5e-5
    dropout = args.drop_rate
    drop_edge = args.drop_rate
    if virtual_node == False:
        drop_edge = 0
    att_dim = 64
    
    # perturbation
    perturbation = args.perturbation
    p_id = 9
    p_std = 1e0
    p_hop = 6
    
    for v in [True, False]:
        for p in [True, False]:
            if p == True:
                for p_std in [1e0, 1e-1, 1e-2 ,1e-3, 1e-4]:
                    for p_hop in [3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
                        print("-"*10 + "{}".format(p_hop) + "-"*10)
                        run(n=run_times, dataset=dataset, virtual_node=v, vn_num=vn_num,
                            initialize=True, num_layers=num_layers, semantic_dim=semantic_dim,
                            att_dim=att_dim, dropout=dropout, lr=lr, weight_decay=weight_decay, drop_edge=drop_edge,
                            perturbation=p, p_id=p_id, p_std=p_std, p_hop=p_hop)
            else:
                for p_hop in [3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
                    print("-"*10 + "{}".format(p_hop) + "-"*10)
                    run(n=run_times, dataset=dataset, virtual_node=v, vn_num=vn_num,
                        initialize=True, num_layers=num_layers, semantic_dim=semantic_dim,
                        att_dim=att_dim, dropout=dropout, lr=lr, weight_decay=weight_decay, drop_edge=drop_edge,
                        perturbation=p, p_id=p_id, p_std=p_std, p_hop=p_hop)
    
    
    # run_savedmodel(n=run_times, dataset=dataset, virtual_node=virtual_node, vn_num=vn_num,
    #     initialize=True, num_layers=num_layers, semantic_dim=semantic_dim,
    #     att_dim=att_dim, dropout=dropout, lr=lr, weight_decay=weight_decay, drop_edge=drop_edge,
    #     perturbation=perturbation, p_id=p_id, p_std=p_std, p_hop=p_hop)