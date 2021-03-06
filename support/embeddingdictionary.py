#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:22:09 2019

@author: student
"""

#complete dictionary and word embedding to all datasets present in data
import io
import numpy as np
class EmbeddingDictionary:
    def read_file(self,file_name):
        ret_list=[]
        fp=open(file_name,'r')
        for line in fp:
            line=line.replace('\n','')
            ret_list.append(line)
        return ret_list    
    
    def create_embedding(self,word2id,embedding_file,dim):
        fin = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        embedding_matrix=np.zeros((len(word2id),dim),dtype=float)
        for line in fin:
            tokens = line.rstrip().split(' ')
            if(tokens[0] in word2id):
                embedding = [float(val) for val in tokens[1:]]
                embedding_matrix[word2id[tokens[0]]] = np.asarray(embedding)
        return embedding_matrix
    def create_cross_embedding(self,word2id,embedding_file,embedding_file_2,dim):
        fin = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        fin2 = io.open(embedding_file_2, 'r', encoding='utf-8', newline='\n', errors='ignore')
        embedding_matrix=np.random.rand(len(word2id),dim)
        for line in fin:
            tokens = line.rstrip().split(' ')
            if(tokens[0] in word2id):
                embedding = [float(val) for val in tokens[1:]]
                embedding_matrix[word2id[tokens[0]]] = np.asarray(embedding)
        for line in fin2:
            tokens = line.rstrip().split(' ')
            if(tokens[0] in word2id):
                embedding = [float(val) for val in tokens[1:]]
                embedding_matrix[word2id[tokens[0]]] = np.asarray(embedding)
        return embedding_matrix
             
    def create_dictionary(self,train):
        word2id={}
        id2word={}
        word2id['OVW']=0
        id2word[0]='OVW'
        index=1
        for line in train:
            for word in line.split():
                if(word not in word2id):   
                    word2id[word]=index
                    id2word[index]=word
                    index=index+1
        word2id['<PAD>']=index
        id2word[index]='<PAD>'
        return word2id,id2word
    def encode_sequence(self,word2id,sent,max_length):    # take a group of sentence and return numpy matrix
        sequence=[]
        for line in sent:
            ret_line=[]
            len_line=max_length-len(line.split())
            for word in line.split():
                if(word in word2id):
                    ret_line.append(word2id[word])
                else:
                    ret_line.append(word2id['OVW'])
            while(len_line>0):
                ret_line.append(word2id['<PAD>'])
                len_line=len_line-1
            sequence.append(ret_line)
        return np.asarray(sequence,dtype='int')
    
    
    
        
        
        
        
        