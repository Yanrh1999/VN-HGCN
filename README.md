# Virtual Nodes based Heterogeneous Graph Network
This is the repository of ICANN2024 paper: [Virtual Nodes based Heterogeneous Graph Convolutional Neural Network for Efficient Long-Range Information Aggregation](https://link.springer.com/chapter/10.1007/978-3-031-72344-5_15). 

DOI: https://doi.org/10.1007/978-3-031-72344-5_15

## Environment
The most relevant package I used:
- Python = 3.8.10
- torch = 2.0.0
- numpy = 1.24.2
- dgl = 1.1.1
- torch-geometric = 2.3.1

If exist error about importing package, please trace the error message to install proper package.

## OpenHGNN
Since code of [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN/) has some issue caused by improperly import the [Networkx](https://networkx.org/) package, I modify it slightly to make it available.
Moreover, Dropedge are adapted for HAN, HetSANN, HGT, ieHGCN and SimpleHGN.
**Before perform VN-HGCN, must installing OpenHGNN,** because the dataset used in VN-HGCN based on this package, and all the baselines also implemented by OpenHGNN. 

Using the commands to install OpenHGNN:
```
cd ./OpenHGNN_pkg
pip install .
```
## ie-HGCN
To validate the effectiveness of adding virtual nodes, perform ie-HGCN and ieHGCN_vn with the commands:
```
cd ./VN-HGCN/ieHGCN
python main.py
```
The optional arguments are same as the [VN-HGCN](#vn-hgcn), except arguments of ```--add_vn, -v``` which decide whether adding virtual nodes, i.e. perform ieHGCN_vn if input `-v True`, else vanilla ieHGCN.  

## VN-HGCN
To perform VN-HGCN, using the commands:
```
cd ./VN-HGCN
python main.py
```
Besides, the optional arguments as follows:

```--dataset, -d```: dataset, expected one of the string in list: ```["acm4NSHE", "dblp4MAGNN","imdb4MAGNN"]```, default value: ```acm4NSHE```.

```--learning_rate, -lr```: learning rate, expected float value, default: ```1e-3```.

```--layers, -n```: number of layers, expected integer value, default: ```4```.

```--vn_num, -vn```: number of virtual nodes on each type, expected integer value, default: ```16```.

> [!WARNING]
> Number of virtual nodes must smaller than number of nodes of any type.

```--times, -t```: number of runs, expected integer value, default: ```5```.

```--train_size, -ts```: draining ratio, expected float value between ```(0, 1)```, default: ```0.4```.

``--drop_rate, -dr``: dropout rate and dropedge rate which are set as the same, expected float value between ```(0, 1)```, default: ```0.5```.

For example, perform VN-HGCN on DBLP dataset:
```
python main.py -d 'dblp4MAGNN' -lr 1e-3 -n 4 -vn 16 -t 5 -ts 0.4
```

If want more advanced option, edit ```main.py``` directly which can adjust the setting including number of central node, the way of initialize virtual node ('ones' or 'mean'), dimension of hidden layer, attention vector and virtual node, weight decay, and set the rate of dropout and dropedge separately.

## Citation
```
@InProceedings{YanNCai-VNHGCN2024,
    author="Yan, Ranhui and Cai, Jia",
    editor="Wand, Michael and Malinovsk{\'a}, Krist{\'i}na and Schmidhuber, J{\"u}rgen and Tetko, Igor V.",
    title="Virtual Nodes based Heterogeneous Graph Convolutional Neural Network for Efficient Long-Range Information Aggregation",
    booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2024",
    year="2024",
    publisher="Springer Nature Switzerland"
}
```