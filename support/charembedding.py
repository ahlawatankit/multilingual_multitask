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
        char2id['<PAD>']=index
        id2char[index]='<PAD>'
        return char2id,id2char
    def create_embedding(self,char2id,dim):
        if len(char2id)>dim:
            dim=len(char2id)
        embedding_matrix=np.zeros((len(char2id),dim),dtype=float)
        for word,index in char2id.items():
            embedding_matrix[index][index]=1.0
        return embedding_matrix
    def encode_sentence(self,char2id,sent,max_char_per_word,max_word_in_line):
        sequence=[]
        for line in sent:
            ret_line=[]
            len_line=max_word_in_line-len(line.split())
            for word in line.split():
                ret_word=[]
                len_word=max_char_per_word-len(word)
                for ch in word:
                    if(ch in char2id):
                        ret_word.append(char2id[ch])
                    else:
                        ret_word.append(char2id['OVC'])
                while(len_word>0):
                    ret_word.append(char2id['<PAD>'])
                    len_word=len_word-1
                ret_line.append(ret_word)
            while(len_line>0):
                ret_word=[]
                for i in range(max_char_per_word):
                    ret_word.append(char2id['<PAD>'])
                ret_line.append(ret_word)
                len_line=len_line-1
            sequence.append(ret_line)
        return np.asarray(sequence,dtype='int')
                
        
