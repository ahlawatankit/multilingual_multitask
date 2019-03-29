#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:34:24 2019

@author: student
"""
# multi lingual multi task trainer
from trainerhelp import Trainerhelp
from support.cleandata import CleanData
from support.embeddingdictionary import EmbeddingDictionary
from support.intent import intent
from support.charembedding import Charembedding
from support.slot import slot
from keras.utils import to_categorical
class MMtrainer:
    def __init__(self,dataset,language1,language2,task,embed_type,model_name,char_level=False):
        self.dataset=dataset
        self.language1=language1
        self.language2=language2
        self.embed_type=embed_type
        self.model_name=model_name
        self.task=task
        self.char_level=char_level
        self.MAX_LENGTH=50
    def  prepare_data(self):
        obj=Trainerhelp(self.dataset,self.language1,self.task,self.embed_type,self.char_level)
        self.train_X_lang1,self.test_X_lang1,self.train_IN_Y_lang1,self.test_IN_Y_lang1,self.train_SO_Y_lang1,self.test_SO_Y_lang1=obj.load_data()
        self.intent2id_lang1,self.id2intent_lang1,self.slot2id_lang1,self.id2slot_lang1=obj.load_label_dict()
        del obj
        obj=Trainerhelp(self.dataset,self.language2,self.task,self.embed_type,self.char_level)
        self.train_X_lang2,self.test_X_lang2,self.train_IN_Y_lang2,self.test_IN_Y_lang2,self.train_SO_Y_lang2,self.test_SO_Y_lang2=obj.load_data()
        self.intent2id_lang2,self.id2intent_lang2,self.slot2id_lang2,self.id2slot_lang2=obj.load_label_dict()
        del obj
        obj=Trainerhelp(self.dataset,self.language1+'_'+self.language2+'_'+self.dataset,self.task,self.embed_type,self.char_level)
        if(self.char_level):
            self.word_embedding,self.char_embedding=obj.load_embedding()
            self.word2id,self.id2word,self.char2id,self.id2char=obj.load_dict()
        else:
            self.word_embedding=obj.load_embedding()
            self.char_embedding=None
            self.word2id,self.id2word=obj.load_dict()
        del obj
        obj=CleanData()
        self.train_X_lang1=obj.adding_tokens(self.train_X_lang1)
        self.test_X_lang1=obj.adding_tokens(self.test_X_lang1)
        self.train_X_lang2=obj.adding_tokens(self.train_X_lang2)
        self.test_X_lang2=obj.adding_tokens(self.test_X_lang2)
        del obj
        
        if self.char_level:
            obj=Charembedding()
            self.train_char_X_lang1=obj.encode_sentence(self.char2id,self.train_X_lang1,25,self.MAX_LENGTH)
            self.test_char_X_lang1=obj.encode_sentence(self.char2id,self.test_X_lang1,25,self.MAX_LENGTH)
            self.train_char_X_lang2=obj.encode_sentence(self.char2id,self.train_X_lang2,25,self.MAX_LENGTH)
            self.test_char_X_lang2=obj.encode_sentence(self.char2id,self.test_X_lang2,25,self.MAX_LENGTH)
        else:
            self.train_char_X_lang1=None
            self.test_char_X_lang1=None
            self.train_char_X_lang2=None
            self.test_char_X_lang2=None
        obj=EmbeddingDictionary()
        self.train_X_lang1=obj.encode_sequence(self.word2id,self.train_X_lang1,self.MAX_LENGTH)
        self.test_X_lang1=obj.encode_sequence(self.word2id,self.test_X_lang1,self.MAX_LENGTH)
        self.train_X_lang2=obj.encode_sequence(self.word2id,self.train_X_lang2,self.MAX_LENGTH)
        self.test_X_lang2=obj.encode_sequence(self.word2id,self.test_X_lang2,self.MAX_LENGTH)
        del obj
        obj=intent()
        self.train_IN_Y_lang1=obj.encode_intent(self.train_IN_Y_lang1,self.intent2id_lang1)
        self.test_IN_Y_lang1=obj.encode_intent(self.test_IN_Y_lang1,self.intent2id_lang1)
        self.train_IN_Y_lang2=obj.encode_intent(self.train_IN_Y_lang2,self.intent2id_lang2)
        self.test_IN_Y_lang2=obj.encode_intent(self.test_IN_Y_lang2,self.intent2id_lang2)
        del obj
        obj=slot()
        self.train_SO_Y_lang1=obj.encode_slot(self.train_SO_Y_lang1,self.slot2id_lang1,self.MAX_LENGTH)
        self.test_SO_Y_lang1=obj.encode_slot(self.test_SO_Y_lang1,self.slot2id_lang1,self.MAX_LENGTH)
        self.train_SO_Y_lang2=obj.encode_slot(self.train_SO_Y_lang2,self.slot2id_lang2,self.MAX_LENGTH)
        self.test_SO_Y_lang2=obj.encode_slot(self.test_SO_Y_lang2,self.slot2id_lang2,self.MAX_LENGTH)
        del obj
        
    def run_model(self):
        if self.model_name=='HCNN':
            from models.multi_language.multi_task.HCNN import HCNN
            obj=HCNN(self.char_embedding,self.word_embedding,len(self.intent2id_lang1),len(self.slot2id_lang1),len(self.intent2id_lang2),len(self.slot2id_lang2),self.dataset,self.language1,self.language2,self.task,self.char_level,self.MAX_LENGTH)
            language_1_model,language_2_model=obj.build_model()
        if self.model_name=='HLSTM':
            from models.multi_language.multi_task.HLSTM import HLSTM
            obj=HLSTM(self.char_embedding,self.word_embedding,len(self.intent2id_lang1),len(self.slot2id_lang1),len(self.intent2id_lang2),len(self.slot2id_lang2),self.dataset,self.language1,self.language2,self.task,self.char_level,self.MAX_LENGTH)
            language_1_model,language_2_model=obj.build_model()
        if self.model_name=='HGRU':
            from models.multi_language.multi_task.HGRU import HGRU
            obj=HGRU(self.char_embedding,self.word_embedding,len(self.intent2id_lang1),len(self.slot2id_lang1),len(self.intent2id_lang2),len(self.slot2id_lang2),self.dataset,self.language1,self.language2,self.task,self.char_level,self.MAX_LENGTH)
            language_1_model,language_2_model=obj.build_model()

        train_loss_IN_lang1,train_accuracy_IN_lang1,train_loss_SO_lang1,train_accuracy_SO_lang1,test_loss_IN_lang1,test_accuracy_IN_lang1,test_loss_SO_lang1,test_accuracy_SO_lang1,max_accuracy_IN_lang1,max_accuracy_SO_lang1,train_loss_IN_lang2,train_accuracy_IN_lang2,train_loss_SO_lang2,train_accuracy_SO_lang2,test_loss_IN_lang2,test_accuracy_IN_lang2,test_loss_SO_lang2,test_accuracy_SO_lang2,max_accuracy_IN_lang2,max_accuracy_SO_lang2=obj.train_model(language_1_model,language_2_model,self.train_X_lang1,self.train_char_X_lang1,to_categorical(self.train_IN_Y_lang1,len(self.intent2id_lang1)),to_categorical(self.train_SO_Y_lang1,len(self.slot2id_lang1)),self.test_X_lang1,self.test_char_X_lang1,to_categorical(self.test_IN_Y_lang1,len(self.intent2id_lang1)),to_categorical(self.test_SO_Y_lang1,len(self.slot2id_lang1)),self.train_X_lang2,self.train_char_X_lang2,to_categorical(self.train_IN_Y_lang2,len(self.intent2id_lang2)),to_categorical(self.train_SO_Y_lang2,len(self.slot2id_lang2)),self.test_X_lang2,self.test_char_X_lang2,to_categorical(self.test_IN_Y_lang2,len(self.intent2id_lang2)),to_categorical(self.test_SO_Y_lang2,len(self.slot2id_lang2)),300)              
        #writing results in ./result
        print("writing results in ./result")
        fp=open('./results/'+self.dataset+'_'+self.language1+'_'+self.language2+'_'+self.task+'_'+self.model_name+'.txt','w')
        fp.write("********Language 1 Intent train Loss and  Accuracy history*******\n")
        fp.writelines(str(train_loss_IN_lang1)+'\n')
        fp.writelines(str(train_accuracy_IN_lang1)+'\n')
        fp.write("******** Language1 Slot train Loss and Accuracy history*******\n")
        fp.writelines(str(train_loss_SO_lang1)+'\n')
        fp.writelines(str(train_accuracy_SO_lang1)+'\n')
        fp.writelines('********Language 1 Intent test loss and accuracy history ***********\n')
        fp.writelines(str(test_loss_IN_lang1)+'\n')
        fp.writelines(str(test_accuracy_IN_lang1)+'\n')
        fp.writelines('********Language 1 Slot test loss and accuracy history ***********\n')
        fp.writelines(str(test_loss_SO_lang1)+'\n')
        fp.writelines(str(test_accuracy_SO_lang1)+'\n')
        
        fp.write("********Language 2 Intent train Loss and  Accuracy history*******\n")
        fp.writelines(str(train_loss_IN_lang2)+'\n')
        fp.writelines(str(train_accuracy_IN_lang2)+'\n')
        fp.write("******** Language2 Slot train Loss and Accuracy history*******\n")
        fp.writelines(str(train_loss_SO_lang2)+'\n')
        fp.writelines(str(train_accuracy_SO_lang2)+'\n')
        fp.writelines('********Language 2 Intent test loss and accuracy history ***********\n')
        fp.writelines(str(test_loss_IN_lang2)+'\n')
        fp.writelines(str(test_accuracy_IN_lang2)+'\n')
        fp.writelines('********Language 2 Slot test loss and accuracy history ***********\n')
        fp.writelines(str(test_loss_SO_lang2)+'\n')
        fp.writelines(str(test_accuracy_SO_lang2)+'\n')
        
        fp.writelines('********language 1 final Intent accuracy***********\n')
        fp.writelines(str(max_accuracy_IN_lang1)+'\n')
        fp.writelines('********language 1 final Slot accuracy***********\n')
        fp.writelines(str(max_accuracy_SO_lang1)+'\n')
        
        fp.writelines('********language 2 final Intent accuracy***********\n')
        fp.writelines(str(max_accuracy_IN_lang2)+'\n')
        fp.writelines('********language 2 final Slot accuracy***********\n')
        fp.writelines(str(max_accuracy_SO_lang2)+'\n')
        fp.close()
if __name__== "__main__":
    print('***Multi Language Multi Task Trainer *********')
    dataset=input("Enter Dataset name ")
    language1=input("Enter language 1 ")
    language2=input("Enter language 2 ")
    task='both'
    embed_type=input("Enter embedding name such as fasttext or word2vec ")
    model_name=input("Enter model name   find in model directory ")
    if(int(input("want to use character embedding press 1 otherwise 0 "))==1):
        char_level=True
    else:
        char_level=False
    obj=MMtrainer(dataset,language1,language2,task,embed_type,model_name,char_level)
    obj.prepare_data()
    obj.run_model()
    del obj
    print("Completed ")           