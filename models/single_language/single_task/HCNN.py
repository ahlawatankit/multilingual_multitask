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
from keras.layers import Dense, Input, Flatten, Dropout, concatenate, Concatenate,BatchNormalization,Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, add, TimeDistributed, Bidirectional, Lambda,Reshape
from evaluate.modelevaluate import ModelEvaluate
from tqdm import tqdm
import math
from keras.optimizers import Adam
CharCNN_config=[[64,4],[64,5],[64,6]]
wordCNN_config=[[256,3],[256,4],[256,5],[256,6]]
UNIT1=64
UNIT2=128
optimizer = Adam(0.0002, 0.5)
losses = 'categorical_crossentropy'
class HCNN:
    def __init__(self,char_embedding,word_embedding,NUM_CLASS,dataset,language,task,CHAR_LEVEL=False,MAX_WORD=90,MAX_CHAR_WORD=18,config_charCNN=CharCNN_config,config_wordCNN=wordCNN_config):
        self.char_embedding=char_embedding
        self.word_embedding=word_embedding
        self.NUM_CLASS=NUM_CLASS
        self.CHAR_LEVEL=CHAR_LEVEL
        self.MAX_WORD=MAX_WORD
        self.MAX_CHAR_WORD=MAX_CHAR_WORD
        self.config_CHARCNN=CharCNN_config
        self.config_wordCNN=wordCNN_config
        self.dataset=dataset
        self.language=language
        self.task=task
        self.model='HCNN'
    def build_model(self):
        #character CNN implemention
        if self.CHAR_LEVEL:
            char_input=Input(shape=(self.MAX_WORD,self.MAX_CHAR_WORD,),dtype='float32',name='Chars_input')
            embed_out=TimeDistributed(Embedding(self.char_embedding.shape[0],self.char_embedding.shape[1],weights=[self.char_embedding],trainable=True))(char_input)
            char_CNN=[]
            for num_filter,filter_size in self.config_charCNN:
                 ch=TimeDistributed(Conv1D(num_filter, filter_size, padding='same',kernel_initializer='random_uniform'))(embed_out)
                 ch=BatchNormalization()(ch)
                 ch=Activation('relu')(ch)
                 char_CNN.append(TimeDistributed(MaxPooling1D(pool_size=2))(ch))
            char_con=concatenate(char_CNN,axis=-1)
            char_out=TimeDistributed(Flatten())(char_con)
        #Word CNN implementation
        word_input = Input(shape=(self.MAX_WORD,), dtype='int32',name='words_input')
        embed_word_out = Embedding(self.word_embedding.shape[0], self.word_embedding.shape[1], weights=[self.word_embedding],trainable=True, mask_zero = False)(word_input)
        if self.CHAR_LEVEL:
            embed_word_out=concatenate([char_out,embed_word_out],axis=-1)
        
        CNN_list=[]
        for num_filter,filter_size in self.config_wordCNN:
           x=Conv1D(num_filter, filter_size,padding='same',kernel_initializer='random_uniform')(embed_word_out)
           x=BatchNormalization()(x)
           x=Activation('relu')(x)
           CNN_list.append(MaxPooling1D(pool_size=2)(x))
        concat_in=concatenate(CNN_list,axis=-1)
        
        if self.task=='intent':
            flat_in=Flatten()(concat_in)
            intent_out = Dense(units=UNIT2, activation='relu', kernel_initializer='he_normal')(flat_in)
            intent_out=Dropout(0.1)(intent_out)
            intent_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='intent_out')(intent_out)
            if self.CHAR_LEVEL:
                graph = Model(inputs=[char_input,word_input], outputs=intent_out)
            else:
                graph = Model(inputs=word_input, outputs=intent_out)
            graph.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
            graph.summary()
            return graph
        elif self.task=='slot':
            slot_in=Reshape((self.MAX_WORD,512))(concat_in)
            slot_out = Dense(units=UNIT2, activation='relu', kernel_initializer='he_normal')(slot_in)
            slot_out=Dropout(0.1)(slot_out)
            slot_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='slot_out')(slot_out)
            if self.CHAR_LEVEL:
                graph = Model(inputs=[char_input,word_input], outputs=slot_out)
            else:
                graph = Model(inputs=word_input, outputs=intent_out)
            graph.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
            graph.summary()
            return graph
            
            
    def train_model(self,graph,X_word,X_char,Y,epochs=50,batch_size=64):
        _loss=[]
        _accuracy=[]
        for ep in tqdm(range(epochs)):
            kf=ModelSupport.get_minibatches_id('',X_word.shape[0],batch_size)
            losses=0.0
            accuracy=0.0
            for bt,idx in kf:
                if self.CHAR_LEVEL:
                    X_train_char=X_char[idx]
                X_train_word=X_word[idx]
                Y_train=Y[idx]
                if self.CHAR_LEVEL:
                    l,a=graph.train_on_batch([X_train_char,X_train_word],Y_train)
                else:
                    l,a=graph.train_on_batch(X_train_word,Y_train)
                losses=losses+l
                accuracy=accuracy+a
            batches=math.ceil(X_word.shape[0] // batch_size)
            losses=losses/batches
            accuracy=accuracy/batches
            if(len(_accuracy)>0 and _accuracy[len(_accuracy)-1]< accuracy):
                graph.save('./models/single_language/single_task/HCNN.h5')
            print('loss after ',ep,'epochs is ',losses)
            print('Training Acuracy ',accuracy)
            _loss.append(losses)
            _accuracy.append(accuracy)
        return _loss,_accuracy
    def test_model(self,graph,X_word,X_char,Y):
        if self.CHAR_LEVEL:
            pred=graph.predict([X_char,X_word])
            pred=np.argmax(pred,axis=1)
            return ModelEvaluate.accuracy('',Y,pred),ModelEvaluate.f1('',Y,pred)
        else:
            pred=graph.predict(X_word)
            pred=np.argmax(pred,axis=1)
            return ModelEvaluate.accuracy('',Y,pred),ModelEvaluate.f1('',Y,pred)
            
        
        

        