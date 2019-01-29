#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:10:47 2019

@author: student
"""
import numpy as np
class Charembedding:
    def create_dict(self,sent):
        id2char = {}
        char2id={}
        char2id['OVC']=0
        id2char[0]='OVC'         #out of vocal charcater
        index=1
        for line in sent:
            for ch in line:
                if(ch not in char2id):
                    char2id[ch]=index
                    id2char[index]=ch
                    index=index+1
        return char2id,id2char
    def create_embedding(self,char2id,dim):
        embedding_matrix=np.zeros((len(char2id),dim),dtype=float)
        for word,index in char2id.items():
            embedding_matrix[index][index]=1.0
        return embedding_matrix