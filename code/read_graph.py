
import json
import torch 
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
def read_hgraph_e_k():
   
    with open('dataset/junyi/graph/raw/Hpgraph_e_k.json','r',encoding='utf-8')as f:
        data = json.load(f)
    
    for i in range(len(data[0])):data[1][i] -= 835
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    edge_index_T = [data[1], data[0]]
    edge_index_T = torch.LongTensor(edge_index_T)
    data = HeteroData()
    data['exer'].num_nodes = 835
    data['knowledge_code'].num_nodes = 835
    edge_index = torch.LongTensor(edge_index)
    data['exer', 'investigate', 'knowledge_code'].edge_index = edge_index
  
    data_T = HeteroData()
    data_T['exer'].num_nodes = 835
    data_T['knowledge_code'].num_nodes = 835
    data_T['knowledge_code', 'investigated', 'exer'].edge_index = edge_index_T
    
    return data,data_T

def read_hgraph_s_k():
    with open('dataset/junyi/graph/raw/HpGraph_u_k.json','r',encoding='utf-8')as f:
        data = json.load(f)
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    edge_index_T = [data[1], data[0]]
    edge_index_T = torch.LongTensor(edge_index_T)
    data = HeteroData()
    data['user'].num_nodes = 10000
    data['knowledge_code'].num_nodes = 835
    edge_index = torch.LongTensor(edge_index)
    data['user', 'master', 'knowledge_code'].edge_index = edge_index

    
    data_T = HeteroData()
    data_T['user'].num_nodes = 10000
    data_T['knowledge_code'].num_nodes = 835
    data_T['knowledge_code', 'examed', 'user'].edge_index = edge_index_T
    return data, data_T

def read_sgraph_s_s():
    with open('dataset/junyi/graph/raw/Smgraph_u_u_15_4W.json','r',encoding='utf-8')as f:
        data = json.load(f)
    
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    data = Data(edge_index=edge_index)
    data.num_nodes = 10000
    return data


def read_sgraph_e_e():
    with open('dataset/junyi/graph/raw/Smgraph_e_e.json','r',encoding='utf-8')as f:
        data = json.load(f)
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    data = Data(edge_index=edge_index)
    data.num_nodes = 835
    return data

def read_sgraph_directed_k_k():
    with open('dataset/junyi/graph/raw/Smgraph_directed_k_k.json','r',encoding='utf-8')as f:
        data = json.load(f)
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    data = Data(edge_index=edge_index)
    data.num_nodes = 835
    return data


def read_sgraph_undirected_k_k():
    with open('dataset/junyi/graph/raw/Smgraph_undirected_k_k.json','r',encoding='utf-8')as f:
        data = json.load(f)
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    data = Data(edge_index=edge_index)
    data.num_nodes = 835
    return data

def read_sgraph_s_e():
    with open('dataset/junyi/graph/raw/u_e.json','r',encoding='utf-8')as f:
        data = json.load(f)
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    
    data = HeteroData()
    data['user'].num_nodes = 10000
    data['exer'].num_nodes = 835
    edge_index = torch.LongTensor(edge_index)
    data['user', 'practice', 'exer'].edge_index = edge_index
    return data

def read_target_s_e(name):
    with open('dataset/junyi/graph/raw/'+name+'_u_e_target_.json','r',encoding='utf-8')as f:
        data = json.load(f)
    edge_index = data
    edge_index = torch.LongTensor(edge_index)
    
    data = HeteroData()
    data['user'].num_nodes = 10000
    data['exer'].num_nodes = 835
    edge_index = torch.LongTensor(edge_index)
    data['user', 'performance', 'exer'].edge_index = edge_index
    return data