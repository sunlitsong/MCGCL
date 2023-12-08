import torch
torch.cuda.current_device()
from torch import nn
from torch.nn import Module

import math
from Attention import wAttention,sAttention
from dataset import MyDataset
from torch_geometric.nn import GCNConv,HypergraphConv,GraphSAGE
from torch_sparse import matmul, mul
import torch_geometric



class HyperConv(Module):
    def __init__(self, layers,dataset,emb_size=1024):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
        item_embeddings = torch.sum(torch.stack(final), 0) / (self.layers+1)
        return item_embeddings

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.linear2 = nn.Sigmoid()
 
    def forward(self, input):
        return self.linear2(self.relu(input))

class MCGCL(Module):
    def __init__(self,args):
        super(MCGCL, self).__init__()
        
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.stu_num = args.student_n
        self.exer_num = args.exer_n
        self.knowledge_num = args.knowledge_n
        self.emb_size = args.emb_size
        self.beta = args.beta
        self.gama = args.gama
        self.layers = args.layers
        self.L2 = args.l2
        self.lr = args.lr   
        
        self.data =  MyDataset('dataset/junyi/graph')
        self.HpGraph_e_k, self.HpGraph_k_e, self.HpGraph_s_k, self.HpGraph_k_s, self.SmGraph_s_s, self.SmGraph_e_e, self.SmGraph_directed_k_k, self.SmGraph_undirected_k_k, self.SmGraph_s_e = self.data[0]
        self.HpGraph_e_k, self.HpGraph_k_e, self.HpGraph_s_k, self.HpGraph_k_s, self.SmGraph_s_s, self.SmGraph_e_e, self.SmGraph_directed_k_k, self.SmGraph_undirected_k_k, self.SmGraph_s_e = self.HpGraph_e_k.to(self.device), self.HpGraph_k_e.to(self.device), self.HpGraph_s_k.to(self.device), self.HpGraph_k_s.to(self.device), self.SmGraph_s_s.to(self.device), self.SmGraph_e_e.to(self.device), self.SmGraph_directed_k_k.to(self.device), self.SmGraph_undirected_k_k.to(self.device), self.SmGraph_s_e.to(self.device)
        self.student_emb = nn.Embedding(self.stu_num, self.emb_size)
        self.exercise_emb = nn.Embedding(self.exer_num, self.emb_size)
        self.knowledge_emb = nn.Embedding(self.knowledge_num, self.emb_size)

        self.k_index = torch.LongTensor(list(range(self.knowledge_num))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.stu_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_num))).to(self.device)
        self.HyperGraph_edge_s_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        self.HyperGraph_edge_s_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        self.HyperGraph_node_s_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        self.HyperGraph_node_s_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        self.HyperGraph_edge_e_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        self.HyperGraph_edge_e_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        self.HyperGraph_node_e_k1 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        self.HyperGraph_node_e_k2 = HypergraphConv(in_channels=self.emb_size, out_channels=self.emb_size,use_attention=True, dropout=0.5)
        
        self.LineGraph_undirect_k = GraphSAGE(in_channels=self.emb_size,hidden_channels=self.emb_size, num_layers=self.layers, out_channels=self.emb_size)
        self.LineGraph_direct_k = GraphSAGE(in_channels=self.emb_size,hidden_channels=self.emb_size, num_layers=self.layers, out_channels=self.emb_size)
        self.LineGraph_e = GraphSAGE(in_channels=self.emb_size, hidden_channels=self.emb_size, num_layers=self.layers,out_channels=self.emb_size)
        self.LineGraph_s = GraphSAGE(in_channels=self.emb_size,hidden_channels=self.emb_size, num_layers=self.layers, out_channels=self.emb_size)

        self.Attention_e = wAttention()
        self.Attention_s = sAttention()
        self.LN = nn.LayerNorm(self.emb_size).to(self.device)

        self.FC1 = nn.Linear(self.emb_size,self.emb_size)
        self.FC2 = nn.Linear(self.emb_size,self.emb_size)
        self.FC3 = nn.Linear(args.emb_size , 128)
        
        self.FC4 = nn.Linear(self.emb_size*2 ,self.emb_size)
        self.FC5 = nn.Linear(self.emb_size*2 ,self.emb_size)
        self.FC6 = nn.Linear(self.emb_size ,self.emb_size)
        self.FC7 = nn.Linear(128 ,1)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.diff=nn.Linear(args.emb_size , 1)
        self.disc=nn.Linear(args.emb_size , args.emb_size)
        self.mlp = MLP(args.emb_size,1)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
     
    def generate_sess_emb(self,item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select


    def score(self, x1, x2):
        return torch.matmul(x1, x2)

    def SSL_e(self,  e_emb_lgcn, e_emb_hgnn):
        lens = e_emb_lgcn.size(0)
        score = self.score(e_emb_lgcn, e_emb_hgnn.transpose(1,0))
        pos = torch.diag(score) 
        neg1 = score-torch.diag_embed(pos)
        one = torch.FloatTensor(neg1.shape[0]).fill_(1).to(self.device)
        con_loss = torch.mean(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def forward(self, stu_id, exer_id, kn_r,batch_cnt=0):
        all_stu_emb = self.student_emb(self.stu_index)
        exer_emb = self.exercise_emb(self.exer_index)
        kn_emb = self.knowledge_emb(self.k_index)

        s_embeddings_hg_s_k = self.HyperGraph_edge_s_k1(x=all_stu_emb, hyperedge_index=self.HpGraph_s_k['user','master','knowledge_code'].edge_index,hyperedge_attr=kn_emb)
        s_embeddings_hg_s_k = self.HyperGraph_edge_s_k2(x=s_embeddings_hg_s_k, hyperedge_index=self.HpGraph_s_k['user','master','knowledge_code'].edge_index,hyperedge_attr=kn_emb)
        s_embeddings_hg_s_k = torch.relu(s_embeddings_hg_s_k)

        edge_index = self.HpGraph_s_k['user','master','knowledge_code'].edge_index[[1,0]]
        k_embeddings_hg_s_k = self.HyperGraph_node_s_k1(x=kn_emb,hyperedge_index=edge_index,hyperedge_attr=s_embeddings_hg_s_k)
        k_embeddings_hg_s_k = self.HyperGraph_node_s_k2(x=k_embeddings_hg_s_k,hyperedge_index=edge_index,hyperedge_attr=s_embeddings_hg_s_k)
        k_embeddings_hg_s_k = torch.relu(k_embeddings_hg_s_k)
        

        e_embeddings_hg_e_k = self.HyperGraph_edge_e_k1(x=exer_emb, hyperedge_index=self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index,hyperedge_attr=kn_emb)
        e_embeddings_hg_e_k = self.HyperGraph_edge_e_k2(x=e_embeddings_hg_e_k, hyperedge_index=self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index,hyperedge_attr=kn_emb)
        e_embeddings_hg_e_k = torch.relu(e_embeddings_hg_e_k)
        edge_index = self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index[[1,0]]
        k_embeddings_hg_e_k = self.HyperGraph_node_e_k1(x=kn_emb,hyperedge_index=edge_index,hyperedge_attr=e_embeddings_hg_e_k)
        k_embeddings_hg_e_k = self.HyperGraph_node_e_k2(x=k_embeddings_hg_e_k,hyperedge_index=edge_index,hyperedge_attr=e_embeddings_hg_e_k)
        k_embeddings_hg_e_k = torch.relu(k_embeddings_hg_e_k)

        k_embeddings_sg_k_k_direction = self.LineGraph_direct_k(kn_emb, self.SmGraph_directed_k_k.edge_index)
        k_embeddings_sg_k_k_undirection = self.LineGraph_undirect_k(kn_emb, self.SmGraph_undirected_k_k.edge_index)
        
        e_embeddings_sg_e_e = self.LineGraph_e(exer_emb, self.SmGraph_e_e.edge_index)
        s_embeddings_sg_s_s = self.LineGraph_s(all_stu_emb, self.SmGraph_s_s.edge_index)
       
        k_emb1 = self.FC4(torch.concat((k_embeddings_hg_s_k,k_embeddings_hg_e_k),dim=-1))
        k_emb2 = self.FC5(torch.concat((k_embeddings_sg_k_k_direction,k_embeddings_sg_k_k_undirection),dim=-1))
        k_embeddings_sg_k_k = self.LN(torch.add(k_emb1,k_emb2))

        adj_e_k = torch.Tensor(torch_geometric.utils.to_scipy_sparse_matrix(self.HpGraph_e_k['exer', 'investigate', 'knowledge_code'].edge_index).toarray()).to(self.device)
        
        exer_embeddings_sg = self.Attention_e(e_embeddings_sg_e_e, k_embeddings_sg_k_k, adj_e_k)
        exer_embeddings_sg = torch.sigmoid(self.FC6(exer_embeddings_sg))

        edge_shape= torch.zeros((self.stu_num,self.knowledge_num)).shape
        value = torch.FloatTensor(torch.ones(self.HpGraph_s_k['user','master', 'knowledge_code'].num_edges)).to(self.device)
        adj_s_k = torch.sparse_coo_tensor(self.HpGraph_s_k['user','master', 'knowledge_code'].edge_index,value,edge_shape).to_dense()
        stu_embeddings_sg = self.Attention_s(s_embeddings_sg_s_s, k_embeddings_sg_k_k,adj_s_k)
        stu_embeddings_sg = torch.sigmoid(stu_embeddings_sg)
        
        #loss of ss
        con_loss_e = self.SSL_e(exer_embeddings_sg, e_embeddings_hg_e_k)
        con_loss_s = self.SSL_e(stu_embeddings_sg,s_embeddings_hg_s_k)

        #SM EM
        exer_emb = torch.tanh(self.FC1(torch.add(exer_embeddings_sg,e_embeddings_hg_e_k)))
        user_emb = torch.tanh(self.FC2(torch.add(stu_embeddings_sg,s_embeddings_hg_s_k)))
        if batch_cnt == 1:
            print("saving embedding")
            torch.save(exer_emb, 'dataset/exer_emb.pt')
            torch.save(user_emb, 'dataset/user_emb.pt')

        batch_stu_emb = user_emb[stu_id]
        batch_exer_emb = exer_emb[exer_id]
        batch_stu_vector = batch_stu_emb.reshape(batch_stu_emb.shape[0],  batch_stu_emb.shape[1])
        batch_exer_vector = batch_exer_emb.reshape(batch_exer_emb.shape[0],  batch_exer_emb.shape[1])

        diff = self.diff(batch_exer_vector)
        disc = self.disc(batch_exer_vector)
        e = diff.view(-1) + torch.sum(torch.mul(batch_stu_vector, disc), 1)
        res = self.mlp(e)
   
        return res.view(len(res), -1), self.beta*con_loss_e, self.gama*con_loss_s