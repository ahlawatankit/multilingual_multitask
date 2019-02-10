# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 15:19:14 2019

@author: HP
"""
from support.embeddingdictionary import EmbeddingDictionary
import numpy as np
import os
class intent:
    def read_intent(self,dataset_name,lang):
        intent= EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset_name+'/'+lang+'/intent_train.txt'))
        intent.extend(EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset_name+'/'+lang+'/intent_test.txt')))
        return intent
    def build_dict(self,intent):
        intent2id={}
        id2intent={}
        index=0
        for inte in intent:
            if(inte not in intent2id):
                intent2id[inte]=index
                id2intent[index]=inte
                index=index+1
        return intent2id,id2intent
    def encode_intent(self,intent,intent2id):
        en_intent=[]
        for inte in intent:
            en_intent.append(intent2id[inte])
        return np.asarray(en_intent,dtype='int32')
    def decode_intent(self,seq_intent,id2intent):
        intent=[]
        for inte in seq_intent:
            intent.append(id2intent[inte])
        return intent
            
        