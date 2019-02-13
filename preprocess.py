#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:45:31 2019

@author: student
"""
from support.embeddingdictionary import EmbeddingDictionary
from support.cleandata import CleanData
from support.charembedding import Charembedding
from support.intent import intent
import os 
import numpy as np
class preprocess:
    def read_all(self,language):
        sent= EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/ATIS/'+language+'/utterence_train.txt'))
        sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/ATIS/'+language+'/utterence_test.txt')))
        sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Frames_data/'+language+'/utterence_train.txt')))
        sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Frames_data/'+language+'/utterence_test.txt')))
        sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Trains_dataset/'+language+'/utterence_train.txt')))
        sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Trains_dataset/'+language+'/utterence_test.txt')))
        return sent
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
    
    def save_dict(self,file,location,file_name):
        np.save(os.path.abspath(location+file_name),file)
    def save_embedding(self,embedding,location):
        np.save(os.path.abspath(location+'embedding.npy'),embedding)
        
         


obj=preprocess()
#####building intent dictionary and saving ###########
intent_obj=intent()
intents=intent_obj.read_intent('ATIS','english')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/ATIS/english/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/ATIS/english/','id2intent.npy')
intents=intent_obj.read_intent('Frames_data','english')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/Frames_data/english/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/Frames_data/english/','id2intent.npy')
intents=intent_obj.read_intent('Trains_dataset','english')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/Trains_dataset/english/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/Trains_dataset/english/','id2intent.npy')
############end########################


###### building for english  ######
sent=obj.read_all('english')
embd,embd_char,word2id,id2word,char2id,id2char=obj.dict_embed(sent,'cc.en.300.vec',300,100)
print('*** Saving English dictionary ***** ')
obj.save_dict(word2id,'./dictionary/english/wordlevel/','word2id.npy')
obj.save_dict(id2word,'./dictionary/english/wordlevel/','id2word.npy')    
obj.save_dict(char2id,'./dictionary/english/charlevel/','char2id.npy')
obj.save_dict(id2char,'./dictionary/english/charlevel/','id2char.npy')
print('*** Saving English Embedding *****')
obj.save_embedding(embd,'./embeddings/english/fasttext/')
obj.save_embedding(embd_char,'./embeddings/english/char/')

##### end ########





