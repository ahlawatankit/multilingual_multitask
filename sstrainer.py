# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:44:52 2019

@author: HP
"""
# single task single language trainer 

from support.embeddingdictionary import EmbeddingDictionary
import os
class SSTrainer:
    def __init__(self,dataset,language,task):
        self.dataset==dataset
        self.language=language
        self.task=task
    def load_data(self):
        train_sent=EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/utterence_train.txt'))
        test_sent=EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/utterence_test.txt'))
        
        if self.task=='intent':
            train_Y=EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/intent_train.txt'))
            test_Y=EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/intent_test.txt'))
            return train_sent,test_sent,train_Y,test_Y
        if self.task=='slot':
            train_Y=EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/slot_train.txt'))
            test_Y=EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/slot_test.txt'))
            return train_sent,test_sent,train_Y,test_Y
    def prepare_data(self,)
        