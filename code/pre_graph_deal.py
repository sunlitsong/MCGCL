import json
from collections import defaultdict

config_file = '/config.txt'
with open(config_file) as i_f:
    i_f.readline()
    student_n, exercise_n, knowledge_n = i_f.readline().split(',')
    student_n, exercise_n, knowledge_n = int(student_n), int(exercise_n), int(knowledge_n)

def s_e():
    edge_target = [[] for _ in range(3)]
    edge_s_e = [[] for _ in range(2)]
    file = 'dataset/junyi/log_data.json'
    with open(file, encoding='utf8') as i_f:
        data = json.load(i_f)

    #train
    for i,item in enumerate(data):
        if i==0:
            edge_s_e[0] = []
            edge_s_e[1] = []
        user =  item['user_id']
        for j,jtem in enumerate(item['logs']):
            exer = jtem['exer_id']
            edge_s_e[0].append(user-1)
            edge_s_e[1].append(exer-1)
    
    with open('./dataset/junyi/graph/u_e.json','w',encoding='utf-8')as f:
        json.dump(edge_s_e,f)
   

def metapath_e_e():
    print('____________________________________________')
    
    e_k_file =  'dataset/junyi/meta_graph/k_from_e.txt'
    k_e_file = 'dataset/junyi/meta_graph/e_from_k.txt'
    
    edge_index = [[] for _ in range(2)]
    k_e_dict = defaultdict(list)
    with open(k_e_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            k_e_dict[int(line[0])].append(int(line[1]))

    with open(e_k_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            for j in k_e_dict[int(line[1])]:
                edge_index[0].append(int(line[0]))
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(int(line[0]))  
    
    with open('./dataset/junyi/graph/Smgraph_e_e.json','w',encoding='utf-8')as f:
        json.dump(edge_index,f)
    

def metapath_u_u():
    print('____________________________________________')
    
    edge_index = [[]for _ in range(2)]
    with open('xx.json','r',encoding='utf-8')as f:
        users = json.load(f)
    print('_____________________________________')

    for i in range(10000):
        for j in range(10000):
            if users[i][j] > 60:
                edge_index[0].append(i)
                edge_index[1].append(j)
    print(len(edge_index[0]))
    import pdb;pdb.set_trace()
    with open('dataset/junyi/graph/raw/Smgraph_u_u_60_8w.json','w',encoding='utf-8')as f: json.dump(edge_index,f)


def new_metapath_u_u():
    edge_index = [[]for _ in range(2)]
    with open('dataset/junyi/graph/raw/count_U_U_same_score.json','r',encoding='utf-8')as f:
        data = json.load(f)
    for i in range(10000):
        for j in range(i,10000):
            if data[i][j] >= 30: # 25:7646
                edge_index[0].append(i)
                edge_index[1].append(j)
    print(len(edge_index[0]))

def metapath_k_k():
    print('____________________________________________')
    
    k_Undirected_file = 'dataset/junyi/meta_graph/K_Undirected.txt'
    edge_Undirected = [[] for _ in range(2)]
    
    with open(k_Undirected_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            
            edge_Undirected[0].append(int(line[0]))
            edge_Undirected[1].append(int(line[1]))
            edge_Undirected[0].append(int(line[1]))
            edge_Undirected[1].append(int(line[0]))  
    
    with open('./dataset/junyi/graph/Smgraph_undirected_k_k.json','w',encoding='utf-8')as f:
        json.dump(edge_Undirected,f)
  

def HyperGraph_e_k():
    print('____________________________________________')
    
    e_k_file =  'dataset/junyi/meta_graph/k_from_e.txt'
    
    edge_index = [[] for _ in range(2)]
    with open(e_k_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_index[0].append(int(line[0]))
            edge_index[1].append(int(line[1]))
   
    with open('./dataset/junyi/graph/Hpgraph_e_k.json','w',encoding='utf-8')as f:
        json.dump(edge_index,f)


def HyperGraph_s_k():
    file = 'dataset/junyi/log_data.json'
    with open(file, encoding='utf8') as i_f:
        train_data = json.load(i_f)
    
    edge_index = [0 for _ in range(2)]

    for i,item in enumerate(train_data):
       
        if i==0:
            edge_index[0] = []
            edge_index[1] = []
        user =  item['user_id']
        for j,jtem in enumerate(item['logs']):
            knowledge = jtem['knowledge_code'][0]
            edge_index[0].append(user-1)
            edge_index[1].append(knowledge-1)
        
    with open('dataset/junyi/graph/HpGraph_u_k.json','w',encoding='utf-8')as f:
        json.dump(edge_index,f)