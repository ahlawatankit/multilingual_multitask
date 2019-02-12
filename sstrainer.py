# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:33:31 2019

@author: HP
"""
from trainerhelp import Trainerhelp
from support.cleandata import CleanData
from support.embeddingdictionary import EmbeddingDictionary
from support.intent import intent
from support.charembedding import Charembedding
from keras.utils import to_categorical
import numpy as np
class SStrainer:
    def __init__(self,dataset,language,task,embed_type,model_name,char_level=False):
        self.dataset=dataset
        self.language=language
        self.task=task
        self.embed_type=embed_type
        self.model_name=model_name
        self.char_level=char_level
    def prepare_data(self):
        obj=Trainerhelp(self.dataset,self.language,self.task,self.embed_type,self.char_level)
        self.train_X,self.test_X,self.train_Y,self.test_Y=obj.load_data()
        if(self.char_level):
            self.word_embedding,self.char_embedding=obj.load_embedding()
            self.word2id,self.id2word,self.char2id,self.id2char=obj.load_dict()
        else:
            self.word_embedding=obj.load_embedding()
            self.char_embedding=None
            self.word2id,self.id2word=obj.load_dict()
        self.task2id,self.id2task=obj.load_label_dict()
        del obj
        obj=CleanData()
        self.train_X=obj.adding_tokens(self.train_X)
        self.test_X=obj.adding_tokens(self.test_X)
        del obj
        # char senetence encoding
        if self.char_level:
            obj=Charembedding()
            self.train_char_X=obj.encode_sentence(self.char2id,self.train_X,18,90)
            self.test_char_X=obj.encode_sentence(self.char2id,self.test_X,18,90)
        # word sentence encoding
        obj=EmbeddingDictionary()
        self.train_X=obj.encode_sequence(self.word2id,self.train_X,90)
        self.test_X=obj.encode_sequence(self.word2id,self.test_X,90)
        del obj
        
        if self.task=='intent':
            obj=intent()
            self.train_Y=obj.encode_intent(self.train_Y,self.task2id)
            self.test_Y=obj.encode_intent(self.test_Y,self.task2id)
            del obj
            
    def run_model(self):
        if self.model_name=='HCNN':
            from models.single_language.single_task.HCNN import HCNN
            obj=HCNN()
            graph=obj.build_model(self.char_embedding,self.word_embedding,len(self.task2id),self.char_level)
            loss,accuracy=obj.train_model(graph,self.train_X,self.train_char_X,to_categorical(self.train_Y,len(self.task2id)))              
            return loss,accuracy
        
        