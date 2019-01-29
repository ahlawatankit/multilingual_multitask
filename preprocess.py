#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:45:31 2019

@author: student
"""
from support.embeddingdictionary import EmbeddingDictionary
from support.cleandata import CleanData
from support.charembedding import Charembedding
import os 
import numpy as np
class preprocess:
    def read_all(self,language):
        english= EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/ATIS/'+language+'/utterence_train.txt'))
        english.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/ATIS/'+language+'/utterence_test.txt')))
        english.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Frames_data/'+language+'/utterence_train.txt')))
        english.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Frames_data/'+language+'/utterence_test.txt')))
        english.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Trains_dataset/'+language+'/utterence_train.txt')))
        english.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Trains_dataset/'+language+'/utterence_test.txt')))
        return english
    def dict_embed(self,sent,embedding_file,dim1,dim2):
         print('*** Parsing Data ****')
         sent1=CleanData.adding_tokens(self,sent)
         print('*** Building Vocab ***')
         word2id,id2word=EmbeddingDictionary.create_dictionary(self,sent1)
         char2id,id2char= Charembedding.create_dict(self,sent)
         print('*** Making Emnbedding ***')
         embedding=EmbeddingDictionary.create_embedding(self,word2id,os.path.abspath(embedding_file),dim1)
         char_embedding=Charembedding.create_embedding(self,char2id,dim2)
         return embedding,char_embedding,word2id,id2word,char2id,id2char
    def save_dict(self,word2id,id2word,location):
        np.save(os.path.abspath(location+'word2id.npy'),word2id)
        np.save(os.path.abspath(location+'id2word.npy'),id2word)
    def save_dict_char(self,char2id,id2char,location):
        np.save(os.path.abspath(location+'char2id.npy'),char2id)
        np.save(os.path.abspath(location+'id2char.npy'),id2char)
    def save_embedding(self,embedding,location):
        np.save(os.path.abspath(location+'embedding.npy'),embedding)
        
         



###### building for english  ######
obj=preprocess()
sent=obj.read_all('english')
embd,embd_char,word2id,id2word,char2id,id2char=obj.dict_embed(sent,'cc.en.300.vec',300,100)
print('*** Saving English dictionary ***** ')
obj.save_dict(word2id,id2word,'./dictionary/english/wordlevel/')      
obj.save_dict_char(char2id,id2char,'./dictionary/english/charlevel/')
print('*** Saving English Embedding *****')
obj.save_embedding(embd,'./embeddings/english/fasttext/')
obj.save_embedding(embd_char,'./embeddings/english/char/')
