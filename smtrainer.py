# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:33:31 2019

@author: HP
"""
import sys
import os
from trainerhelp import Trainerhelp
from support.cleandata import CleanData
from support.embeddingdictionary import EmbeddingDictionary
from support.intent import intent
from support.charembedding import Charembedding
from support.slot import slot
from keras.utils import to_categorical

import numpy as np
class SMtrainer:
    def __init__(self,dataset,language,task,embed_type,model_name,char_level=False):
        self.dataset=dataset
        self.language=language
        self.task=task
        self.embed_type=embed_type
        self.model_name=model_name
        self.char_level=char_level
        self.MAX_LENGTH=50
    def prepare_data(self):
        obj=Trainerhelp(self.dataset,self.language,self.task,self.embed_type,self.char_level)
        self.train_X,self.test_X,self.train_IN_Y,self.test_IN_Y,self.train_SO_Y,self.test_SO_Y=obj.load_data()
        if(self.char_level):
            self.word_embedding,self.char_embedding=obj.load_embedding()
            self.word2id,self.id2word,self.char2id,self.id2char=obj.load_dict()
        else:
            self.word_embedding=obj.load_embedding()
            self.char_embedding=None
            self.word2id,self.id2word=obj.load_dict()
        self.intent2id,self.id2intent,self.slot2id,self.id2slot=obj.load_label_dict()
        del obj
        obj=CleanData()
        self.train_X=obj.adding_tokens(self.train_X)
        self.test_X=obj.adding_tokens(self.test_X)
        del obj
        # char senetence encoding
        if self.char_level:
            obj=Charembedding()
            self.train_char_X=obj.encode_sentence(self.char2id,self.train_X,25,self.MAX_LENGTH)
            self.test_char_X=obj.encode_sentence(self.char2id,self.test_X,25,self.MAX_LENGTH)
        else:
            self.train_char_X=None
            self.test_char_X=None
        # word sentence encoding
        obj=EmbeddingDictionary()
        self.train_X=obj.encode_sequence(self.word2id,self.train_X,self.MAX_LENGTH)
        self.test_X=obj.encode_sequence(self.word2id,self.test_X,self.MAX_LENGTH)
        del obj
        obj=intent()
        self.train_IN_Y=obj.encode_intent(self.train_IN_Y,self.intent2id)
        self.test_IN_Y=obj.encode_intent(self.test_IN_Y,self.intent2id)
        del obj
        obj=slot()
        self.train_SO_Y=obj.encode_slot(self.train_SO_Y,self.slot2id,self.MAX_LENGTH)
        self.test_SO_Y=obj.encode_slot(self.test_SO_Y,self.slot2id,self.MAX_LENGTH)
        del obj
    def run_model(self):
        if self.model_name=='HCNN':
            from models.single_language.multi_task.HCNN import HCNN
            obj=HCNN(self.char_embedding,self.word_embedding,len(self.intent2id),len(self.slot2id),self.dataset,self.language,self.task,self.char_level,self.MAX_LENGTH)
        if self.model_name=='HLSTM':
            from models.single_language.multi_task.HLSTM import HLSTM
            obj=HLSTM(self.char_embedding,self.word_embedding,len(self.intent2id),len(self.slot2id),self.dataset,self.language,self.task,self.char_level,self.MAX_LENGTH)
        if self.model_name=='HGRU':
            from models.single_language.multi_task.HGRU import HGRU
            obj=HGRU(self.char_embedding,self.word_embedding,len(self.intent2id),len(self.slot2id),self.dataset,self.language,self.task,self.char_level,self.MAX_LENGTH)

        graph=obj.build_model()
        train_loss_IN,train_accuracy_IN,train_loss_SO,train_accuracy_SO,test_loss_IN,test_accuracy_IN,test_loss_SO,test_accuracy_SO,max_accuracy_IN,max_accuracy_SO=obj.train_model(graph,self.train_X,self.train_char_X,to_categorical(self.train_IN_Y,len(self.intent2id)),to_categorical(self.train_SO_Y,len(self.slot2id)),self.test_X,self.test_char_X,to_categorical(self.test_IN_Y,len(self.intent2id)),to_categorical(self.test_SO_Y,len(self.slot2id)),250)              
        #writing results in ./result
        print("writing results in ./result")
        fp=open('./results/'+self.dataset+'_'+self.language+'_'+self.task+'_'+self.model_name+'.txt','w')
        fp.write("******** Intent train Loss and  Accuracy history*******\n")
        fp.writelines(str(train_loss_IN)+'\n')
        fp.writelines(str(train_accuracy_IN)+'\n')
        fp.write("******** Slot train Loss and Accuracy history*******\n")
        fp.writelines(str(train_loss_SO)+'\n')
        fp.writelines(str(train_accuracy_SO)+'\n')
        fp.writelines('********Intent test loss and accuracy history ***********\n')
        fp.writelines(str(test_loss_IN)+'\n')
        fp.writelines(str(test_accuracy_IN)+'\n')
        fp.writelines('********Slot test loss and accuracy history ***********\n')
        fp.writelines(str(test_loss_SO)+'\n')
        fp.writelines(str(test_accuracy_SO)+'\n')
        fp.writelines('********final Intent accuracy***********\n')
        fp.writelines(str(max_accuracy_IN)+'\n')
        fp.writelines('********final Slot accuracy***********\n')
        fp.writelines(str(max_accuracy_SO)+'\n')
        fp.close()
if __name__== "__main__":
    print('***Single Language Multi Task Trainer *********')
    dataset=input("Enter Dataset name ")
    language=input("Enter language ")
    task='both'
    embed_type=input("Enter embedding name such as fasttext or word2vec ")
    model_name=input("Enter model name   find in model directory ")
    if(int(input("want to use character embedding press 1 otherwise 0 "))==1):
        char_level=True
    else:
        char_level=False
    obj=SMtrainer(dataset,language,task,embed_type,model_name,char_level)
    obj.prepare_data()
    obj.run_model()
    del obj
    print("Completed ")        
        
