#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:22:09 2019

@author: student
"""

#complete dictionary and word embedding to all datasets present in data
import datetime
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
        embedding_matrix=np.zeros((len(word2id)+1,300),dtype=float)
        for line in fin:
            tokens = line.rstrip().split(' ')
            if(tokens[0] in word2id):
                embedding = [float(val) for val in tokens[1:]]
                embedding_matrix[word2id[tokens[0]]] = np.asarray(embedding)
        return embedding_matrix
             
    def create_dictionary(self,train):
        word2id={}
        id2word={}
        index=1
        for line in train:
            for word in line.split():
                if(word not in word2id):   
                    word2id[word]=index
                    id2word[index]=word
                    index=index+1
        return word2id,id2word
        
        
        
        