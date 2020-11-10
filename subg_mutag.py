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

def glrt_init(shape, name=None): ########initialization function#######
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def activation_layer(act,input,name=None): #######activation functions##########
    if act=="sigmoid":
        layer = tf.nn.sigmoid(input,name=name)
        return layer
    elif act=="relu":
        layer = tf.nn.relu(input,name=name)
        return layer
    elif act=="swish":
        layer = tf.nn.swish(input,name=name)
        return layer
    elif act=="tanh":
        layer = tf.nn.tanh(input,name=name)
        return layer
    elif act=="leaky_relu":

        layer = tf.nn.leaky_relu(input,name=name)
        return layer

def Subgraph_Feature_Transform(d,train,A,in_feat,out_feat,reuse):
##############This function transforms the featres to new features.##########
    with tf.variable_scope("transforms", reuse=False):
        # A = tf.nn.dropout(A, rate=d)
        init_range = np.sqrt(6.0/(in_feat+out_feat))
        initial = tf.random_uniform([in_feat,out_feat], minval=-init_range, maxval=init_range, dtype=tf.float32)
        weight= tf.Variable(initial, name="weight_1")
        H1=tf.tensordot(A,weight, axes=[[-1], [0]])
    return H1

def Subgraph_Attention(d,train,A,in_feat,out_feat,reuse):
##########This function calculates the attention scores.##########
    with tf.variable_scope("transform_attention", reuse=False):
        init_range = np.sqrt(6.0/(in_feat+out_feat))
        initial = tf.random_uniform([in_feat,1], minval=-init_range, maxval=init_range, dtype=tf.float32)
        weight= tf.Variable(initial, name="weight_2")
        H1=tf.tensordot(A,weight,axes=[[-1], [0]])
    return H1

def GCN_layer(d,A1,H,out_feat,in_feat,act,name='gcn',i=0,k=1,train=True):#k1=0.3):
    weights = glrt_init([in_feat, out_feat],name=name)
    n12=A1.get_shape().as_list()
    eps=tf.Variable(tf.zeros(1))
    # shape=[in_feat,1]
    # init_range = np.sqrt(6.0/(shape[0]+shape[1]))  
    # A=A1+tf.eye(n12[1])#,batch_shape=n12[0])  
    # A=A1+tf.eye(n12[1])#,batch_shape=n12[0])  
    rowsum = tf.reduce_sum(A1,axis=2)            
    d_inv_sqrt = tf.contrib.layers.flatten(tf.rsqrt(rowsum))
    d_inv_sqrt=tf.where(tf.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)        
    d_inv_sqrt=tf.linalg.diag(d_inv_sqrt)       
    A1=tf.matmul(tf.matmul(d_inv_sqrt,A1),d_inv_sqrt)
    H=tf.nn.dropout(H,1-d)#,training=k)
    A1=tf.matmul(A1,H,name=name+'matmul1')
    A1=A1+(1+eps)*H 

    for i in range(2-1):
      ad=tf.keras.layers.Dense(units = out_feat)(A1)
      ab=tf.keras.layers.BatchNormalization()(ad)
      input_features = tf.nn.relu(ab)

    H_next=tf.keras.layers.Dense(units = out_feat)(input_features)
    ab=tf.keras.layers.BatchNormalization()(H_next)
    H_next = tf.nn.relu(ab)

    # H_next=tf.tensordot(A1, weights, axes=[[2], [0]])
    # H_next=activation_layer(act,H_next,name=name+'matmul1')
    return H_next

def GCN_layer1(d,A1,H,out_feat,in_feat,act,name='gcn',i=0,k=1,train=True):   
    weights = glrt_init([in_feat, out_feat],name=name)
    n12=A1.get_shape().as_list()
    eps=tf.Variable(tf.zeros(1))
    # shape=[in_feat,1]
    # init_range = np.sqrt(6.0/(shape[0]+shape[1]))  
    # A=A1+tf.eye(n12[1])#,batch_shape=n12[0])  
    # A=A1+tf.eye(n12[1])#,batch_shape=n12[0])  
    rowsum = tf.reduce_sum(A1,axis=2)            
    d_inv_sqrt = tf.contrib.layers.flatten(tf.rsqrt(rowsum))
    d_inv_sqrt=tf.where(tf.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)        
    d_inv_sqrt=tf.linalg.diag(d_inv_sqrt)       
    A1=tf.matmul(tf.matmul(d_inv_sqrt,A1),d_inv_sqrt)
    H=tf.layers.dropout(H,rate=d,training=k)
    A1=tf.matmul(A1,H,name=name+'matmul1')
    A1=A1+(1+eps)*H  
    H_next=tf.tensordot(A1, weights, axes=[[2], [0]])
    H_next=activation_layer(act,H_next,name=name+'matmul1')
    return H_next

# def GCN_layer(d,A,H,out_feat,in_feat,act,name='gcn',i=0,k=1,train=True):
# ########Perfroms node_neighhood information aggregation############
#     weights = glrt_init([in_feat, out_feat],name=name)
#     n12=A.get_shape().as_list()
#     # # shape=[in_feat,1]
#     # # init_range = np.sqrt(6.0/(shape[0]+shape[1]))  
#     A=A+tf.eye(n12[1],batch_shape=n12[0])  
#     rowsum = tf.reduce_sum(A,axis=2)            
#     d_inv_sqrt = tf.contrib.layers.flatten(tf.rsqrt(rowsum))
#     # print("d",d_inv_sqrt)
#     # d_inv_sqrt=tf.where(tf.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt) 
#     # print("d",d_inv_sqrt)
#     d_inv_sqrt=tf.linalg.diag(d_inv_sqrt)       
#     A=tf.matmul(tf.matmul(d_inv_sqrt,A),d_inv_sqrt)  
#     H=tf.layers.dropout(H,rate=d,training=train)
# #     H=tf.layers.dropout(H,rate=d,training=train)    
#     A=tf.matmul(A,H,name=name+'matmul1')  
#     H=tf.tensordot(A, weights, axes=[[2], [0]])
#     H=activation_layer(act,H,name=name+'matmul1')
#     return H

class SubGattPool(object):
    def __init__(self,placeholders):
        self.num_pool=placeholders['num_pool1']
        self.num_nodes=placeholders['num_nodes1']
        self.out_feat1=placeholders['outp_dim1']
        self.clusratio=placeholders['clusrstio']
        self.nclasses=placeholders['nclasses1']
        self.learning_rate=placeholders['learning_rate1']
        self.subgraphsamp=placeholders['samples']
        self.subgraphlength=placeholders['subgraph_length']
        self.subgatt_layers=placeholders['subgatt_layers1']
        self.feat_dim=placeholders['feat_dim1']
        self.acts="relu"

    def Subgraph_Attention_Layer(self,feat,lay): 
        ############This is one SubGatt Layer################
        for layers in range(self.subgatt_layers):
            zer=tf.zeros([1,feat.get_shape().as_list()[2]],tf.float32)
            feat1=tf.map_fn(lambda x: tf.concat((x, zer), axis=0), feat)
            with tf.variable_scope("trans", reuse=False): 
                subgraphs21=tf.gather_nd(feat1,self.subgraphs)  
                subgraphs21=tf.reshape(subgraphs21,[-1,subgraphs21.get_shape().as_list()[1],subgraphs21.get_shape().as_list()[2],subgraphs21.get_shape().as_list()[3]*subgraphs21.get_shape().as_list()[4]])                          
                # print("------subgraphs21------------",subgraphs21)
                subgraphs21=Subgraph_Feature_Transform(self.drop_rate,self.train1,subgraphs21, subgraphs21.get_shape().as_list()[3], lay,reuse=False) #features transformation done...        
                s2=Subgraph_Attention(self.drop_rate,self.train1,subgraphs21,subgraphs21.get_shape().as_list()[3],1,reuse=False) 
                             
                s2=tf.nn.leaky_relu(s2)              
                s2=tf.nn.softmax((tf.reshape(s2,[-1,s2.get_shape().as_list()[1],s2.get_shape().as_list()[2]])),axis=-1)
                # print("------",s2)             
                s2=tf.reshape(s2,[-1,s2.get_shape().as_list()[1],1,s2.get_shape().as_list()[2]]) 
                feat=tf.matmul(s2,subgraphs21) 
                feat=tf.reshape(feat,[-1,feat.get_shape().as_list()[1],feat.get_shape().as_list()[3]])  
                feat=tf.math.l2_normalize(feat,2)
        return feat


    def Emb_Pooling_layer(self,clusnext,d,A,x,out_feat,in_feat,act,link_pred,j,i):

        ############This is one SubGattPool Layer################
        p=(A.shape[2])
        with tf.variable_scope("node_gnn",reuse=False):
            if i==0:
                z_l=x
                for i1 in range(1):
                    z_l=self.Subgraph_Attention_Layer(z_l,out_feat)
                z_l1=x
                in_feat=z_l1.get_shape().as_list()[2] 
                for i1 in range(0,1):                
                    z_l1=self.Subgraph_Attention_Layer(z_l1,clusnext)
                x_l1=tf.matmul(tf.transpose((tf.nn.softmax(z_l1,axis=-1)),[0,2,1]),z_l)  
                x_l1=tf.math.l2_normalize(x_l1,2)
                A_l1=tf.matmul(tf.matmul(tf.transpose((tf.nn.softmax(z_l1,axis=-1)),[0,2,1]),A),(tf.nn.softmax(z_l1,axis=-1)))
            else:
                z_l=x
                for i1 in range(2):
                    z_l=GCN_layer(d,A,z_l,out_feat,in_feat,act,i=j,train=self.train1)
                    in_feat=z_l.get_shape().as_list()[2]
                z_l1=x
                in_feat=z_l1.get_shape().as_list()[2] 
                for i1 in range(0,3):
                   # print(clusnext,in_feat)
                    z_l1=GCN_layer(d,A,z_l1,clusnext,in_feat,act,i=j,train=self.train1)
                    in_feat=z_l1.get_shape().as_list()[2]
                x_l1=tf.matmul(tf.transpose((tf.nn.softmax(z_l1,axis=-1)),[0,2,1]),z_l)  
                x_l1=tf.math.l2_normalize(x_l1,2)
                A_l1=tf.matmul(tf.matmul(tf.transpose((tf.nn.softmax(z_l1,axis=-1)),[0,2,1]),A),(tf.nn.softmax(z_l1,axis=-1)))

        return x_l1,A_l1


    def Intra_present_level_attention(self,H,A,out_feat,name=None):
    #########Intra layer component(Explained in paper) ################
        in_feat=H.get_shape().as_list()[-1]
        shape=[in_feat,1]
        init_range = np.sqrt(6.0/(shape[0]+shape[1]))        
        with tf.variable_scope("node_gnn111",reuse=tf.compat.v1.AUTO_REUSE):
            initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
            weights = tf.get_variable(initializer=initial, name="ww")  
            n12=A.get_shape().as_list()
            # shape=[in_feat,1]
            # init_range = np.sqrt(6.0/(shape[0]+shape[1]))  
            A=A+tf.eye(n12[1])#,batch_shape=n12[0])             
            rowsum = tf.reduce_sum(A,axis=2)            
            d_inv_sqrt = tf.contrib.layers.flatten(tf.rsqrt(rowsum))
            d_inv_sqrt=tf.where(tf.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)        
            d_inv_sqrt=tf.linalg.diag(d_inv_sqrt)       
            A_l1=tf.matmul(tf.matmul(d_inv_sqrt,A),d_inv_sqrt)
            
        A=tf.matmul(A_l1,H)
        H_next=tf.tensordot(A, weights, axes=[[-1], [0]])
        H_next=tf.nn.softmax(H_next,axis=-2)
        H_next=tf.matmul(tf.transpose(H_next,[0,2,1]),H)        
        return H_next
       
    def Inter_present_level_attention(self,H,name=None):
    ##########Inter layer Component (explained in paper)###########
        in_feat=H.get_shape().as_list()[-1]
        weights = glrt_init([in_feat, 1],name=name)
        with tf.variable_scope("node_gnn211",reuse=False):
            H_next=tf.tensordot(H, weights, axes=[[-1], [0]])  
        
        
        H_next=tf.nn.softmax(H_next,axis=-2)
        H_next=tf.matmul(tf.transpose(H_next,[0,2,1]),H)
        return H_next
        
        
    def Classifier(self,input1):
    ##############A Classifier ON THE TOP OF EMBEDDINGS LEARNED BY SubGattPool architechture##################
        initializer = tf.contrib.layers.xavier_initializer
        node_feat_rec = tf.layers.dense(input1, self.nclasses, activation=None, use_bias=False,kernel_initializer=initializer())
        return node_feat_rec
    
    def Calc_Loss(self, logits, labels,reuse=False):
        l2_reg_lambda=1e-5
        with tf.name_scope('loss'):
            lossL2=0
#             self.loss_val=(tf.losses.softmax_cross_entropy(labels,logits,label_smoothing=0.01))
            self.loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            self.Optimizer(self.loss_val)#+lossL2)
    
    def Model_Architechture(self):
    ##############COMPLETE ARCHITECHTURE OF THE PAPER################
        self._add_placeholders()
        A=self.input_adj
        x=self.input_features
        in_feat=x.get_shape().as_list()
        in_feat=in_feat[2]
        out_feat=self.out_feat1
        link_pred=True
        x=GCN_layer(self.drop_rate,A,x,out_feat,in_feat,self.acts,i=1,train=self.train1)
        in_feat=x.get_shape().as_list()[2]
        with tf.variable_scope('arch'):
            clusnext=int(self.num_nodes * self.clusratio)
            for i in range(self.num_pool): 
                if i==self.num_pool-1:
                    x_l1,A_l1=self.Emb_Pooling_layer(1,self.drop_rate,A,x,out_feat,x.get_shape().as_list()[2],self.acts,link_pred,1,i)
                else:               
                    x_l1,A_l1=self.Emb_Pooling_layer(clusnext,self.drop_rate,A,x,out_feat,x.get_shape().as_list()[2],self.acts,link_pred,0,i)
                    Z=self.Intra_present_level_attention(x_l1,A_l1,1)
                    if i==0:
                        z=Z 
                    else:
                        z=tf.concat([z,Z],axis=1)
                A=A_l1
                x=x_l1
                in_feat=x.get_shape().as_list()[2]
                clusnext=int(self.clusratio*clusnext)
                if clusnext==0 and i <(self.num_pool-2):
                    print(clusnext,i)
                    raise Exception('Either reduce pooling ratio or number of SubGatt pooling layers as #nodes becoming zero for next layer')
        z=tf.concat([z,x],axis=1)   
        x=self.Inter_present_level_attention(z)
        z1=tf.reshape(x,[-1,x.get_shape().as_list()[2]])
        output=self.Classifier(tf.reshape(x,[-1,x.get_shape().as_list()[2]]))
        labels=self.input_labels  
        # self.Calc_Loss(output,labels)
        # y_pred=output        
        y_pred =tf.nn.softmax(output)
        y_pred_cls = tf.argmax(y_pred, dimension=1,name='pp')         
        l=tf.argmax(labels, dimension=1,name="pp1")
        correct_prediction = tf.equal(y_pred_cls, l)       
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="mean")       
        self.Calc_Loss(output,labels)
        

    def runn(self,sess,feed1,v):
        feed={self.train1:feed1['train1'],self.subgraphs:feed1['sub'],self.drop_rate:feed1['keep'],self.input_features:feed1['input_features1'],self.input_adj:feed1['input_adj1'],self.input_labels:feed1['input_labels1']}
        if v=="train":            
            run_vars = [tf.trainable_variables(),self.train_op_np]
            t,c = sess.run(run_vars, feed_dict = feed)
            run_vars=[tf.get_default_graph().get_tensor_by_name("Reshape:0"),tf.get_default_graph().get_tensor_by_name("mean:0"),self.loss_val]
            emd,a,summ = sess.run(run_vars, feed_dict = feed)
            return t,a,summ


        
        elif v=="val" or v=="test":
            run_vars = [ 
                        tf.get_default_graph().get_tensor_by_name("arch/node_gnn/trans/Reshape:0"),                    
                        tf.get_default_graph().get_tensor_by_name("pp1:0"),
                        tf.get_default_graph().get_tensor_by_name("pp:0"),
                        tf.get_default_graph().get_tensor_by_name("mean:0"),
                        self.loss_val]
            aa,kk,y_predd,a,summ = sess.run(run_vars, feed_dict = feed)
            return kk,y_predd,a,summ,aa
     

    def Optimizer(self,reuse=False):
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        with tf.name_scope('opt'):
            self.learning_rate = tf.train.exponential_decay(self.learning_rate , global_step, 100000, 0.96, staircase=True)
            self.train_op_np = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val) #, global_step = self.global_ste
     

    def _add_placeholders(self):
        self.train1=tf.placeholder(tf.bool)
        self.input_features = tf.placeholder(tf.float32, shape = [None, self.num_nodes,self.feat_dim], name = "input_features")
        self.input_adj = tf.placeholder(tf.float32, shape = [None,self.num_nodes, self.num_nodes], name = "input_adj")
        self.input_labels = tf.placeholder(tf.int32, shape = [None, self.nclasses], name = "input_labels")           
        self.drop_rate = tf.placeholder(tf.float32)
        self.subgraphs = tf.placeholder(tf.int32, shape = [None, self.num_nodes,self.subgraphsamp,self.subgraphlength,2], name = "subgraphs")
