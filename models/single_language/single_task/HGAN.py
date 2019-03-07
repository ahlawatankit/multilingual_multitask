#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:30:43 2019

@author: student
"""
from support.modelsupport import ModelSupport
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, concatenate,BatchNormalization,Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, add, TimeDistributed, Bidirectional, Lambda,Reshape
from evaluate.modelevaluate import ModelEvaluate
from tqdm import tqdm
from keras.optimizers import Adam
CharCNN_config=[[32,4],[32,5],[32,6]]
wordCNN_config=[[128,3],[128,4],[128,5],[128,6]]
UNIT1=64
UNIT2=128
optimizer = Adam(lr=0.0001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.00009, amsgrad=False)
losses = 'categorical_crossentropy'
class HGAN:
    def __init__(self,char_embedding,word_embedding,NUM_CLASS,dataset,language,task,CHAR_LEVEL=False,MAX_WORD=90,MAX_CHAR_WORD=25,lstm_unit=512,config_charCNN=CharCNN_config,config_wordCNN=wordCNN_config):
        self.char_embedding=char_embedding
        self.word_embedding=word_embedding
        self.NUM_CLASS=NUM_CLASS
        self.CHAR_LEVEL=False#CHAR_LEVEL
        self.MAX_WORD=MAX_WORD
        self.MAX_CHAR_WORD=MAX_CHAR_WORD
        self.config_charCNN=CharCNN_config
        self.config_wordCNN=wordCNN_config
        self.lstm_unit=lstm_unit
        self.dataset=dataset
        self.language=language
        self.task=task
        self.model='HGAN' 
        self.n_embedding=self.noise_mix(self.word_embedding)
    def noise_mix(self,embedding):
        # elemnt wise multiplication with normal distribution
        n_embedding=[]
        for vec in embedding:
            e=np.random.normal(0,1,embedding.shape[1])
            new_vec=np.multiply(vec,e)
            n_embedding.append(new_vec)
            #print(np.linalg.norm(vec),np.linalg.norm(new_vec))
        return np.asarray(n_embedding)
    def build_discriminator(self):
         #character CNN implemention
        word_input = Input(shape=(self.MAX_WORD,self.word_embedding.shape[1]), dtype='float32',name='words_input')
        CNN_list=[]
        for num_filter,filter_size in self.config_wordCNN:
           x=Conv1D(num_filter, filter_size,padding='same',kernel_initializer='random_uniform')(word_input)
           x=BatchNormalization()(x)
           x=Activation('relu')(x)
           CNN_list.append(MaxPooling1D(pool_size=2)(x))
        concat_in=concatenate(CNN_list,axis=-1)
        
        if self.task=='intent':
            flat_in=Flatten()(concat_in)
            feature_out = Dense(units=UNIT2, activation='relu', kernel_initializer='he_normal')(flat_in)
            feature_out=Dropout(0.3)(feature_out)
            intent_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='intent_out')(feature_out)
            valid_out=Dense(units=1, activation='sigmoid', kernel_initializer='he_normal',name ='valid_out')(feature_out)
            discriminator = Model(inputs=word_input, outputs=[intent_out,valid_out])
            discriminator.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
            discriminator.summary()
            return discriminator
        elif self.task=='slot':
            slot_in=Reshape((self.MAX_WORD,256))(concat_in)
            slot_out = Dense(units=UNIT2, activation='relu', kernel_initializer='he_normal')(slot_in)
            slot_out=Dropout(0.3)(slot_out)
            feature_out=Flatten()(concat_in)
            valid_out=Dense(units=1, activation='sigmoid', kernel_initializer='he_normal',name ='valid_out')(feature_out)
            slot_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='slot_out')(slot_out)
            discriminator = Model(inputs=word_input, outputs=[slot_out,valid_out])
            discriminator.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
            
            discriminator.summary()
            return discriminator
    def build_generator(self):
        word_input = Input(shape=(self.MAX_WORD,self.word_embedding.shape[1]), dtype='float32',name='words_input')
        lstm_out=LSTM(self.lstm_unit, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='ones', unit_forget_bias=True, kernel_regularizer=keras.regularizers.l2(0.01), dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(word_input)
        bt=BatchNormalization()(lstm_out)
        lstm_out=LSTM(self.lstm_unit, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='ones', unit_forget_bias=True, kernel_regularizer=keras.regularizers.l2(0.01), dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(bt)
        bt=BatchNormalization()(lstm_out)
        generator = Model(inputs=word_input, outputs=bt)
        generator.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
        generator.summary()
        return generator
    def build_model(self):
        self.discriminator=self.build_discriminator()
        self.generator = self.build_generator()
        fake_sent = self.generator()
        self.discriminator.trainable = False
        valid, target_label = self.discriminator(fake_sent)
        self.combined = Model(word_input, [valid, target_label])
        self.combined.compile(loss=losses,optimizer=optimizer)
        return self.combined