import tensorflow as tf
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import networkx as nx
import random
import sklearn
import random
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import preprocessing
import collections
import gc
import scipy as sc
import os
import re
import itertools
import statistics
import pickle
import argparse
from subg_mutag import *
# from graph_classification1 import SubGattPool

def one(i,n):
    a = np.zeros(n, 'uint8')
    a[i] = 1
    return a

def subgraph_generation(graph, root,sub_length): 
    ########Generates subgraphs for every node#############
    visited, nodes_in_queu = set(), collections.deque([root])
    subgraphs=[]
    l2=[]
    l2.append(root)
    subgraphs.append(l2)
    present_level={}
    node_parent={}    
    node_parent[root]=-1
    present_level[root]=0
    visited.add(root)
    while nodes_in_queu: 
        present_node = nodes_in_queu.popleft()
        if present_level[present_node]<=sub_length-1:            
            for node_neigh in graph[present_node]: 
                if node_neigh not in present_level:
                    present_level[node_neigh]=present_level[present_node]+1                
                if present_level[node_neigh]<=sub_length-1:                    
                    if node_neigh not in visited: 
                        visited.add(node_neigh) 
                        node_parent[node_neigh]=present_node
                        nodes_in_queu.append(node_neigh) 
                        l1=[]
                        l1.append(node_neigh)
                        p=node_parent[node_neigh]
                        while p!=-1:                       
                            l1.append(p)
                            p=node_parent[p]
                        l2=[]
                        for i in range(len(l1)-1,-1,-1):
                            l2.append(l1[i])
                        subgraphs.append(l2)
                    elif node_neigh in visited and node_neigh in nodes_in_queu:# in visited and node_neigh not in nodes_in_queu and node_neigh==present_node:
                        l1=[]
                        l1.append(node_neigh)
                        p=present_node
                        while p!=-1:                       
                            l1.append(p)
                            p=node_parent[p]
                        l2=[]
                        for i in range(len(l1)-1,-1,-1):
                            l2.append(l1[i])
                        if len(l1)<=sub_length:
                            subgraphs.append(l2)                
        else:
            break   
    return subgraphs


################################ THIS FUNCTION (read_graphfile) IS ADAPTED FROM RexYing/diffpool ############################################


def read_graphfile(dataname, sub_length):
    max_nodes=None
    #read datasets
    prefix='dataset_graph/'+dataname+'/'+dataname
    data_list=[]
    data={}
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
     
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])       
        
        if max_nodes is not None and G.number_of_nodes()> max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = np.array(node_label_one_hot)
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
#             print(node_attrs[0].shape[0])
#             break
            G.graph['feat_dim'] = node_attrs[0].shape[0]
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
       
        graphs.append(nx.relabel_nodes(G, mapping))

    max_num_nodes = max([G.number_of_nodes() for G in graphs])
    lab1=[]

    # feat_dim = graphs[0].node[0]['feat'].shape[0]
    feat_dim1 = graphs[0].node[0]['label'].shape[0]

    for G in graphs:        
        adj = np.array(nx.to_numpy_matrix(G))
        
        neigh=[]
        for i in range(len(adj)):
            l=[]
            for j in range(len(adj)):
                if adj[i][j]>0:
                    l.append(j)
            neigh.append(l)
        gc.collect()     
        
        subgraphs1=[]
        for i in range(len(neigh)):
            l=np.array(subgraph_generation(neigh,i,sub_length))
            l1=[]
            for i1 in range(len(l)):
                l1.append(l[i1])
            subgraphs1.append(l1)
        
   
        num_nodes = adj.shape[0]
        
        
        adj_padded = np.zeros((max_num_nodes,max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        label1=G.graph['label']
        if len(node_attrs)>0:#==G.number_of_nodes():
            f = np.zeros((max_num_nodes,feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['feat']
        else:
            max_deg = 63
            f = np.zeros((max_num_nodes,feat_dim1), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['label']
            # rowsum = np.array(f.sum(1))
            # r_inv = np.power(rowsum, -1).flatten()
            # r_inv[np.isinf(r_inv)] = 0.
            # r_mat_inv = sp.diags(r_inv)
            # f = r_mat_inv.dot(f)
                
            degs = np.sum(np.array(adj), 1).astype(int)

            degs[degs>max_deg] = max_deg
            feat = np.zeros((len(degs), max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1
            feat = np.pad(feat, ((0, max_num_nodes - G.number_of_nodes()), (0, 0)),
                    'constant', constant_values=0)

            f = np.concatenate((feat, f), axis=1)
            f1=np.identity(max_num_nodes)

            rowsum = np.array(f.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            f = r_mat_inv.dot(f)

        lab1.append(label1) 
        label1=one(label1,len(label_vals))
        data={}
        data['feat']=f
        data['adj']=adj_padded
        data['label']=label1
        data['subgraphs1']=subgraphs1
        data_list.append(data)
    return data_list, len(label_vals),max_num_nodes,lab1
 
def glrt_init(shape, name=None): ########initialization function#######
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def Subgraph_Sampling(set_inp,batch_size,max_num_nodes,sub_samples,sub_length):
    ##############Sample subgraphs for each node and for all graphs in every epoch#################
    i10=0
    subgraph=[]
    for i11 in range(len(set_inp)):
        if i10==batch_size:
            i10=0
        subgraphs2=set_inp[i11]
        d1=[]       

        for i3 in range(len(subgraphs2)):
            ind=[i20 for i20 in range(len(subgraphs2[i3]))]                     
            if len(ind)>=sub_samples:                  
                idx=np.random.choice(len(ind),size=sub_samples,replace=False)
            else:
                ind=np.random.choice(len(ind),size=len(ind),replace=False)#list(itertools.permutations(ind))
               # random.shuffle(ind)
                idx=ind
                leng=sub_samples-len(ind)
                while(leng!=0):

                    if leng<len(ind):
                        lis=[ind[i33] for i33 in range(leng)]
                    else:
                        lis=[ind[i33] for i33 in range(len(ind))]
                    leng=leng-len(lis)
                    idx=np.concatenate([idx,lis])

            d2=[]
            for j3 in idx:
                l=subgraphs2[i3][j3]
                l1=np.asarray(l).reshape(1,-1)
                d3=[]
                k=0

                for k in range(l1.shape[1]):
                    d3.append([i10,l1[0][k]])
                if k<sub_length:
                    while k!=sub_length-1:
                        d3.append([i10,max_num_nodes])
                        k+=1
                d2.append(d3)

            d1.append(d2)  
        if i3!=max_num_nodes-1:
            subp=[]
            for jl in range(sub_length):
                subp.append([i10,max_num_nodes])#,[i10,max_num_nodes],[i10,max_num_nodes],[i10,max_num_nodes]]
            while i3!=max_num_nodes-1:
                idx=sub_samples #np.random.randint(len(subgraphs2[i3]), size=int(8))
                d2=[]
                for j3 in range(idx):
                    d2.append(subp)
                d1.append(d2) 
                i3+=1
        subgraph.append(d1)
        i10+=1
    return subgraph


def read_dataset(dataset,subg_length):
    ################### read in graphs ########################
    datasets,n_classes,max_num_node,lab1=read_graphfile(dataset,subg_length)
    datasets=np.array(datasets)
    return datasets,n_classes,max_num_node,lab1

def train_val_test(adj,labels,feat,subgraphs1,arguments,nclasses,max_num_nodes,lab1):
    
    epoch1=arguments.epoch
    sub_length=arguments.sub_leng
    sub_samples=arguments.sub_samp
    pool_layers=arguments.pool_lay
    pool_rate=arguments.pool_rt
    # print("*********************************MMMMMMMMMMM",max_num_nodes)    
    # print(nclasses)
    # print(len(adj))
    ################Training, validation and testing###################
    placeholders={}
    final={}
    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=0) #KFold(n_splits=10)
    vallos=[]
    valf=[]
    valaccr=[]

    testaccr=[]
    testlos=[]
    testfmicro=[]
    testfmacro=[]
    losss=[]
    loss=[]
    ep=[[] for ir in range(10)]
    it=0
    for train_index, test_index in kf.split(adj,lab1):
        train_label,test_label=labels[train_index],labels[test_index]
        train_feat,test_feat=feat[train_index],feat[test_index]
        train_adj,test_adj=adj[train_index],adj[test_index]
        train_subgraphs1,test_subgraphs1=subgraphs1[train_index],subgraphs1[test_index]

                  # placeholders={'l2':0.05,'g':2,'learning_rate':0.0001,'num_nodes':max_num_nodes,'num_pool':3,'gcnlayers':3,'acts':"relu",'feat_dim':feat[0].shape[1],'emb_dim':20,'hid_dim':20,'clusrstio':0.25,'nclasses':nclasses}
        placeholders={'feat_dim1':feat[0].shape[1],'subgraph_length':sub_length,'samples':sub_samples,'subgatt_layers1':arguments.sub_lay,'learning_rate1':arguments.learning_rate,'num_nodes1':max_num_nodes,'num_pool1':pool_layers,'outp_dim1':arguments.embd_dim,'clusrstio':pool_rate,'nclasses1':nclasses}

        a1=0
        pat=20 #patience for early stopping 
        tf.reset_default_graph()
        tf.set_random_seed(123)
        D=SubGattPool(placeholders)
        ######Change batch_size according to the dataset######
        batch_size=len(train_adj)
        num_batches=int(len(train_adj)/batch_size)
        D.Model_Architechture()

        #####to set GPU ##########
        # config = tf.ConfigProto(device_count = {'GPU': 4})
        # config.gpu_options.allow_growth=False

        sess = tf.Session()#config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        subb=[]
        vlss_mn = np.inf
        vacc_mx = 0.0
        asqmx=0.0
        step = 0

        for epoch in range(300):
            trainavrloss = 0
            trainavracc = 0
            vallossavr = 0
            valaccravr = 0
            subgraph=[]
            i10=0
            batch_size=len(train_adj)
            subgraph=Subgraph_Sampling(train_subgraphs1,batch_size,max_num_nodes,sub_samples,sub_length)
            tr_step = 0
            tr_size = len(train_adj)
            # print("epoch",epoch)
            for j in range(num_batches):            
                feed = {}
                feed['input_features1'] = train_feat[j*batch_size:j*batch_size+batch_size]
                feed['input_adj1'] = train_adj[j*batch_size:j*batch_size+batch_size]
                feed['input_labels1']=train_label[j*batch_size:j*batch_size+batch_size]
                feed['sub']=subgraph[j*batch_size:j*batch_size+batch_size]
                feed['keep']=arguments.dropout
                feed['train1']=True
                t,a,summ1=D.runn(sess,feed,"train")
                trainavrloss += summ1
                trainavracc += a
                tr_step += 1          

            feed = {}
            subgraph=[]
            i10=0
            batch_size=len(test_adj)
            subgraph=Subgraph_Sampling(test_subgraphs1,batch_size,max_num_nodes,sub_samples,sub_length)
            
            feed['input_features1'] = test_feat
            feed['input_adj1'] = test_adj
            feed['input_labels1']=test_label
            feed['keep']=0
            feed['sub']=subgraph
            feed['train1']=False

            k1,y,a,summ1,aa=D.runn(sess,feed,"val")
                    
            ep[it].append(a*100)
        print('Epoch',epoch,"Accuracy on train set",trainavracc/tr_step,"test set",a)#"loss",summ)
        it+=1
        # print("************************************************")

    ep1=np.mean(ep,axis=0)
    ep11=ep1.tolist()
    epi=ep11.index(max(ep11))
    print("Best Average Accuracy on all 10 splits",max(ep11))

    

def argument_parser():
    parser = argparse.ArgumentParser(description="SubGattPool for graph classification")
    parser.add_argument("-dt", "--dataset", type=str, help="name of the dataset", default="MUTAG")
    parser.add_argument("-ep", "--epoch", type=int, default=300, help="Number of Epochs")
    # parser.add_argument("-ss", "--sub_samp", type=int, default=12, help="number of subgraphs to be sampled")

    parser.add_argument("-ss", "--sub_samp", type=int, default=12, help="number of subgraphs to be sampled")
    # parser.add_argument("-sl", "--sub_leng", type=int, default=4, help="subgraph length")

    parser.add_argument("-sl", "--sub_leng", type=int, default=3, help="subgraph length")
    parser.add_argument("-pr", "--pool_rt", type=float, default=0.75, help="pooling ratio")
    parser.add_argument("-pl", "--pool_lay", type=int, default=2, help="number of pooling layers")
    parser.add_argument("-sbl", "--sub_lay", type=int, default=2, help="number of SubGatt layers")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-ed", "--embd_dim", type=int, default=128, help="embedding dimension")
    parser.add_argument("-dr", "--dropout", type=float, default=0., help="dropout rate")

    arguments = parser.parse_args()
    return arguments


def main():
    arguments = argument_parser()
    dataset,nclasses,max_num_nodes,lab1=read_dataset(arguments.dataset,arguments.sub_leng)
    #################SEPERATE EACH COMPONENT###################
    adj=[]
    labels=[]
    feat=[]
    subgraphs1=[]
    for i in range(len(dataset)):
        adj.append(dataset[i]['adj'])
        labels.append(dataset[i]['label'])
        feat.append(dataset[i]['feat'])
        subgraphs1.append(dataset[i]['subgraphs1'])
    #print(len(adj),len(labels),len(feat),len(subgraphs1))
    adj=np.array(adj)
    labels=np.array(labels)
    feat=np.array(feat)
    subgraphs1=np.array(subgraphs1)
    lab1=np.array(lab1)
    train_val_test(adj,labels,feat,subgraphs1,arguments,nclasses,max_num_nodes,lab1)
#Main Function
if __name__ == "__main__":
    main()
#hyperparametrs values are setvaccording to the MUTAG dataset in this code.
