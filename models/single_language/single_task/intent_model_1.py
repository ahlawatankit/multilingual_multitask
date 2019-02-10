#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:24:11 2019

@author: student
"""

from support.modelsupport import ModelSupport
import numpy as np
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, concatenate, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, add, concatenate, TimeDistributed, Bidirectional, Lambda,Reshape
from evaluate.modelevaluate import ModelEvaluate
import tqdm
import math
from keras.optimizers import Adam
CharCNN_config=[[32,4],[32,5],[32,6]]
wordCNN_config=[[64,4],[64,5],[64,6]]
UNIT1=64
UNIT2=128
optimizer = Adam(0.0002, 0.5)
losses = 'categorical_crossentropy'
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
        graph.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
        return graph
    def train_model(self,graph,X,Y,epochs=50,batch_size=64):
        X_char=X[0]
        X_word=X[1]
        _loss=[]
        _accuracy=[]
        for ep in tqdm(range(epochs)):
            kf=ModelSupport.get_minibatches_id('',X_char.shape[0],batch_size)
            losses=0.0
            accuracy=0.0
            for bt,idx in kf:
                X_train_char=X_char[idx]
                X_train_word=X_word[idx]
                Y_train=Y[idx]
                l,a=graph.train_on_batch([X_train_char,X_train_word],Y_train)
                losses=losses+l
                accuracy=accuracy+a
            batches=math.ceil(X_char.shape[0] // batch_size)
            losses=losses/batches
            accuracy=accuracy/batches
            if(len(_accuracy)>0 and _accuracy[len(_accuracy)-1]> accuracy):
                graph.save('HCNN.h5')
            _loss.append(losses)
            _accuracy.append(accuracy)
        return _loss,_accuracy
    def test_model(self,graph,Validation_Data):
        pred=graph.predict(Validation_Data[:-1])
        actual=Validation_Data[-1]
        return ModelEvaluate.accuracy('',actual,pred),ModelEvaluate.f1('',actual,pred)
        
        

        