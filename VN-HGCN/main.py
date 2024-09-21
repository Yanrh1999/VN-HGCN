import dgl
import VNHGCN
from pyg2dgl import data_transform
from Add_VN import add_vn
from Data import LoadData
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', default='acm4NSHE', type=str, help='name of datasets')
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--layers', '-n', default=4, type=int, help='number of layers')
    parser.add_argument('--vn_num', '-vn', default=16, type=int, help='number of virtual nodes')
    parser.add_argument('--times', '-t', default=5, type=int, help='number of runs')
    parser.add_argument('--train_size', '-ts', default=.4, type=float, help='training size')
    parser.add_argument('--drop_rate', '-dr', default=.3, type=float, help='dropout rate and dropedge rate')
    args = parser.parse_args()
    
    # dataset _setting
    dataset = args.dataset
    train_size = args.train_size
    random_state = 0
    data_transform(dataset=dataset, train_size=train_size, random_state=random_state)

    # virtual nodes setting
    vn_num = args.vn_num
    vn_dim = 64
    central_vn_num = 1
    vn_initialize = 'mean'
    g = dgl.load_graphs("./dataset/{}.bin".format(dataset))[0][0]
    add_vn(g=g, dataset=dataset, vn_num=vn_num, vn_dim=vn_dim, central_vn_num=central_vn_num,
           vn_initialize=vn_initialize)

    # model setting
    run_times = args.times
    num_layers = args.layers
    hidden_dim = 64
    lr = args.learning_rate
    weight_decay = 1e-4
    dropout = args.drop_rate
    drop_edge = args.drop_rate
    att_dim = 64
    VNHGCN.run(n=run_times, dataset=dataset, initialize=True, num_layers=num_layers, hidden_dim=hidden_dim,
               att_dim=att_dim, dropout=dropout, lr=lr, weight_decay=weight_decay, drop_edge=drop_edge)
