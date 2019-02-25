# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:44:52 2019

@author: HP
"""
# single task single language trainer 

from support.embeddingdictionary import EmbeddingDictionary
import os
import numpy as np
class Trainerhelp:
    def __init__(self,dataset,language,task,embed_type,char_level=False):
        self.dataset=dataset
        self.language=language
        self.task=task
        self.embed_type=embed_type
        self.char_level=char_level
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
    def load_embedding(self):
        if self.language=='english' and self.embed_type=='fasttext':
            word_embed=np.load(os.path.abspath('./embeddings/'+self.language+'/'+self.embed_type+'/embedding.npy'))
            if self.char_level==False:
                return word_embed
            if self.char_level==True:
                char_embed=np.load(os.path.abspath('./embeddings/'+self.language+'/char/embedding.npy'))
                return word_embed,char_embed
    def load_dict(self):
        if self.language=='english':
            word2id=np.load(os.path.abspath('./dictionary/'+self.language+'/wordlevel/word2id.npy')).item()
            id2word=np.load(os.path.abspath('./dictionary/'+self.language+'/wordlevel/id2word.npy')).item()
            if(self.char_level):
                 char2id=np.load(os.path.abspath('./dictionary/'+self.language+'/charlevel/char2id.npy')).item()
                 id2char=np.load(os.path.abspath('./dictionary/'+self.language+'/charlevel/id2char.npy')).item()
                 return word2id,id2word,char2id,id2char
            else:
                return word2id,id2word
    def load_label_dict(self):
        if self.task=='intent':
            intent2id=np.load(os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/intent2id.npy')).item()
            id2intent=np.load(os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/id2intent.npy')).item()
            return intent2id,id2intent
        elif self.task=='slot':
            intent2id=np.load(os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/slot2id.npy')).item()
            id2intent=np.load(os.path.abspath('./datasets/'+self.dataset+'/'+self.language+'/id2slot.npy')).item()
            return intent2id,id2intent

        