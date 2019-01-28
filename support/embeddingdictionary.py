#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:22:09 2019

@author: student
"""

#complete dictionary and embedding to all datasets present in data
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
    
    def create_embedding(self,word2id,embedding_file):
        fin = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        embedding_matrix=np.zeros((len(word2id),len(word2id)),dtype=float)
        for line in fin:
            tokens = line.rstrip().split(' ')
            if(tokens[0] in word2id):
                embedding = [float(val) for val in tokens[1:]]
                embedding[word2id[tokens[0]]] = embedding
        return embedding_matrix
    @staticmethod
    def check_date(word):
        #date formate mm/dd/yy
        date=word.split('/')
        if(len(date)<2 or len(date)>3):
            return word
        if(len(date)==2):
            try :
                datetime.datetime(2015,int(date[0]),int(date[1]))
            except ValueError :
                return word
        if(len(date)==3):
             try :
                datetime.datetime(int(date[2]),int(date[0]),int(date[1]))
             except ValueError :
                return word
        return '#date'
    @staticmethod
    def isFloat(word):
        try:
            float(word)
            return '#num'
        except ValueError:
            return word
    @staticmethod
    def isInt(word):
        try:
            int(word)
            return '#num'
        except ValueError:
            return word
    @staticmethod
    def isTime(word):
        time=word.split(':')
        if(len(time)==2):
            if(EmbeddingDictionary.isInt(time[0]) and EmbeddingDictionary.isInt(time[1])):
                return '#time'
        return word
    
    def adding_tokens(self,sentence):
        sent=[]
        for line in sentence:
            line1=[]
            for word in line:
                word=EmbeddingDictionary.isInt(word)
                word=EmbeddingDictionary.isFloat(word)
                word=EmbeddingDictionary.isTime(word)
                word=EmbeddingDictionary.check_date(word)
                line1.append(word)
            line1=line1   #start here 
            
    def create_dictionary(self,train):
        word2id={}
        id2word={}
        index=1
        for line in train:
            for word in line.split():
                if(word.isnumeric() or EmbeddingDictionary.isFloat(word)):#numeric  and date
                    word='#num'
                if(EmbeddingDictionary.check_date(word)):
                    word='#date'                
                if(word not in word2id):   
                    word2id[word]=index
                    id2word[index]=word
                    index=index+1
        return word2id,id2word
        
        
        
        