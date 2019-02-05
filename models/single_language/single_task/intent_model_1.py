#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:24:11 2019

@author: student
"""
CharCNN_config=[[32,4],[32,5],[32,6]]
wordCNN_config=[[64,4],[64,5],[64,6]]
UNIT1=64
UNIT2=128
import numpy as np
import os
from keras import regularizers
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten, Dropout, concatenate, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, add, concatenate, TimeDistributed, Bidirectional, Lambda,Reshape
from evaluate.modelevaluate import ModelEvaluate
class HCNN:
    def build_model(self,char_embedding,word_embedding,NUM_CLASS,MAX_WORD=100,MAX_CHAR_WORD=26,config_charCNN=CharCNN_config,config_wordCNN=wordCNN_config):
        #character CNN implemention
        char_input=Input(shape=(MAX_WORD,MAX_CHAR_WORD,),dtype='float32',name='Chars_input')
        embed_out=TimeDistributed(Embedding(char_embedding.shape[0],char_embedding.shape[1],weights=[char_embedding],trainable=True))(char_input)
        char_CNN=[]
        for num_filter,filter_size in config_charCNN:
             ch=TimeDistributed(Conv1D(num_filter, filter_size, padding='same',activation='tanh'))(embed_out)
             char_CNN.append(TimeDistributed(MaxPooling1D(pool_size=2))(ch))
        char_con=concatenate(char_CNN,axis=-1)
        char_out=TimeDistributed(Flatten())(char_con)
        #Word CNN implementation
        word_input = Input(shape=(MAX_WORD,), dtype='int32',name='words_input')
        embed_word_out = Embedding(word_embedding.shape[0], word_embedding.shape[1], weights=[word_embedding],trainable=False, mask_zero = False)(word_input)
        embed_word_out=concatenate([char_out,embed_word_out],axis=-1)
        CNN_list=[]
        for num_filter,filter_size in config_wordCNN:
           x=Conv1D(num_filter, filter_size,padding='same',activation='tanh')(embed_word_out)
           CNN_list.append(MaxPooling1D(pool_size=2)(x))
        concat_in=concatenate(CNN_list,axis=-1)
        flat_in=Flatten()(concat_in)
        flat_in=Dropout(0.5)(flat_in)
        intent_out = Dense(units=UNIT1, activation='relu', kernel_initializer='he_normal')(flat_in)
        intent_out=Dropout(0.5)(intent_out)
        intent_out = Dense(units=NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='intent_out')(intent_out)
        graph = Model(inputs=[char_input,word_input], outputs=intent_out)
        return graph
    def train_model(self,graph,model_train_input,model_output,Validation_Data=None,epochs=50,batch_size=64):
        saveBestModel = ModelCheckpoint(filepath="HCNN.hdf5", monitor='val_slot_out_acc', verbose=1, save_best_only=True, mode='auto')
        history=graph.fit( model_train_input,model_output,validation_data=Validation_Data, epochs, batch_size, verbose=2, callbacks=[saveBestModel])
        return history
    def test_model(self,graph,Validation_Data):
        pred=graph.predict(Validation_Data[:-1])
        actual=Validation_Data[-1]
        
        

        