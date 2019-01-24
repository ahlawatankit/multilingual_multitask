#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:22:09 2019

@author: student
"""

#complete dictionary and embedding to all datasets present in data

import os
import io
import numpy as np
class EmbeddingDictionary:
    def __init__(self,pretrain_embedding_file):
        
        
    def create_embedding(self,word2id,embedding_file):
        fin = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        embedding_matrix=np.zeros((len(word2id),len(word2id)),dtype=float)
        for line in fin:
            tokens = line.rstrip().split(' ')
            if(tokens[0] in word2id):
                embedding = [float(val) for val in tokens[1:]]
                embedding[word2id[tokens[0]]] = embedding
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
        
        
        
        