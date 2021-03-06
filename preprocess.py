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
from support.slot import slot
import os 
import numpy as np
class preprocess:
    def read_all(self,language,dataset=None,cross_lingual=False,language_1=None):
        if(cross_lingual):
            sent= EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset+'/'+language+'/utterence_train.txt'))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset+'/'+language+'/utterence_test.txt')))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset+'/'+language_1+'/utterence_train.txt')))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset+'/'+language_1+'/utterence_test.txt')))      
            return sent
        else:   
            sent= EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/ATIS/'+language+'/utterence_train.txt'))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/ATIS/'+language+'/utterence_test.txt')))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Frames_data/'+language+'/utterence_train.txt')))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Frames_data/'+language+'/utterence_test.txt')))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Trains_dataset/'+language+'/utterence_train.txt')))
            sent.extend( EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/Trains_dataset/'+language+'/utterence_test.txt')))
            return sent
    def dict_embed(self,sent,embedding_file,dim1,dim2,embedding_2=None,cross_lingual=False):
         print('*** Parsing Data ****')
         sent1=CleanData.adding_tokens(self,sent)
         print('*** Building Vocab ***')
         word2id,id2word=EmbeddingDictionary.create_dictionary(self,sent1)
         char2id,id2char= Charembedding.create_dict(self,sent)
         print('*** Making Emnbedding ***')
         if(cross_lingual):
              embedding=EmbeddingDictionary.create_cross_embedding(self,word2id,os.path.abspath(embedding_file),os.path.abspath(embedding_2),dim1)
         else:
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

slot_obj=slot()
slots=slot_obj.read_slot('ATIS','english')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/ATIS/english/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/ATIS/english/','id2slot.npy')
slots=slot_obj.read_slot('Frames_data','english')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/Frames_data/english/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/Frames_data/english/','id2slot.npy')
slots=slot_obj.read_slot('Trains_dataset','english')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/Trains_dataset/english/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/Trains_dataset/english/','id2slot.npy')



############hindi##############
intents=intent_obj.read_intent('ATIS','hindi')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/ATIS/hindi/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/ATIS/hindi/','id2intent.npy')
intents=intent_obj.read_intent('Frames_data','hindi')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/Frames_data/hindi/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/Frames_data/hindi/','id2intent.npy')
intents=intent_obj.read_intent('Trains_dataset','hindi')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/Trains_dataset/hindi/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/Trains_dataset/hindi/','id2intent.npy')

slots=slot_obj.read_slot('ATIS','hindi')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/ATIS/hindi/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/ATIS/hindi/','id2slot.npy')
slots=slot_obj.read_slot('Frames_data','hindi')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/Frames_data/hindi/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/Frames_data/hindi/','id2slot.npy')
slots=slot_obj.read_slot('Trains_dataset','hindi')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/Trains_dataset/hindi/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/Trains_dataset/hindi/','id2slot.npy')


############## ben################

intents=intent_obj.read_intent('ATIS','ben')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/ATIS/ben/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/ATIS/ben/','id2intent.npy')
intents=intent_obj.read_intent('Frames_data','ben')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/Frames_data/ben/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/Frames_data/ben/','id2intent.npy')
intents=intent_obj.read_intent('Trains_dataset','ben')
intent2id,id2intent=intent_obj.build_dict(intents)
obj.save_dict(intent2id,'./datasets/Trains_dataset/ben/','intent2id.npy')
obj.save_dict(intent2id,'./datasets/Trains_dataset/ben/','id2intent.npy')
del intent_obj

slots=slot_obj.read_slot('ATIS','ben')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/ATIS/ben/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/ATIS/ben/','id2slot.npy')
slots=slot_obj.read_slot('Frames_data','ben')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/Frames_data/ben/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/Frames_data/ben/','id2slot.npy')
slots=slot_obj.read_slot('Trains_dataset','ben')
slot2id,id2slot=slot_obj.build_dict(slots)
obj.save_dict(slot2id,'./datasets/Trains_dataset/ben/','slot2id.npy')
obj.save_dict(slot2id,'./datasets/Trains_dataset/ben/','id2slot.npy')
del slot_obj
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

#########building for hindi ############
sent=obj.read_all('hindi')
embd,embd_char,word2id,id2word,char2id,id2char=obj.dict_embed(sent,'cc.hi.300.vec',300,100)
print('*** Saving hindi dictionary ***** ')
obj.save_dict(word2id,'./dictionary/hindi/wordlevel/','word2id.npy')
obj.save_dict(id2word,'./dictionary/hindi/wordlevel/','id2word.npy')    
obj.save_dict(char2id,'./dictionary/hindi/charlevel/','char2id.npy')
obj.save_dict(id2char,'./dictionary/hindi/charlevel/','id2char.npy')
print('*** Saving English Embedding *****')
obj.save_embedding(embd,'./embeddings/hindi/fasttext/')
obj.save_embedding(embd_char,'./embeddings/hindi/char/')

##### end ########

#########building for ben ############
sent=obj.read_all('ben')
embd,embd_char,word2id,id2word,char2id,id2char=obj.dict_embed(sent,'cc.bn.300.vec',300,100)
print('*** Saving ben dictionary ***** ')
obj.save_dict(word2id,'./dictionary/ben/wordlevel/','word2id.npy')
obj.save_dict(id2word,'./dictionary/ben/wordlevel/','id2word.npy')    
obj.save_dict(char2id,'./dictionary/ben/charlevel/','char2id.npy')
obj.save_dict(id2char,'./dictionary/ben/charlevel/','id2char.npy')
print('*** Saving English Embedding *****')
obj.save_embedding(embd,'./embeddings/ben/fasttext/')
obj.save_embedding(embd_char,'./embeddings/ben/char/')

##### end ########

#multi lingual part

######### hindi_ben  ############
sent=obj.read_all('hindi','ATIS',True,'ben')
embd,embd_char,word2id,id2word,char2id,id2char=obj.dict_embed(sent,'cross_hi.vec',300,100,'cross_bn.vec',True)
print('*** Saving hindi ben dictionary ***** ')
obj.save_dict(word2id,'./dictionary/hindi_ben_ATIS/wordlevel/','word2id.npy')
obj.save_dict(id2word,'./dictionary/hindi_ben_ATIS/wordlevel/','id2word.npy')    
obj.save_dict(char2id,'./dictionary/hindi_ben_ATIS/charlevel/','char2id.npy')
obj.save_dict(id2char,'./dictionary/hindi_ben_ATIS/charlevel/','id2char.npy')
print('*** Saving hindi  ben Embedding *****')
obj.save_embedding(embd,'./embeddings/hindi_ben_ATIS/fasttext/')
obj.save_embedding(embd_char,'./embeddings/hindi_ben_ATIS/char/')

######### hindi_ben  ############
sent=obj.read_all('hindi','Trains_dataset',True,'ben')
embd,embd_char,word2id,id2word,char2id,id2char=obj.dict_embed(sent,'cross_hi.vec',300,100,'cross_bn.vec',True)
print('*** Saving hindi ben dictionary ***** ')
obj.save_dict(word2id,'./dictionary/hindi_ben_Trains_dataset/wordlevel/','word2id.npy')
obj.save_dict(id2word,'./dictionary/hindi_ben_Trains_dataset/wordlevel/','id2word.npy')    
obj.save_dict(char2id,'./dictionary/hindi_ben_Trains_dataset/charlevel/','char2id.npy')
obj.save_dict(id2char,'./dictionary/hindi_ben_Trains_dataset/charlevel/','id2char.npy')
print('*** Saving hindi  ben Embedding *****')
obj.save_embedding(embd,'./embeddings/hindi_ben_Trains_dataset/fasttext/')
obj.save_embedding(embd_char,'./embeddings/hindi_ben_Trains_dataset/char/')

del obj