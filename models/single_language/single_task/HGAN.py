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
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM,Reshape
from evaluate.modelevaluate import ModelEvaluate
from tqdm import tqdm
from keras.optimizers import Adam,SGD
from keras.utils import to_categorical
CharCNN_config=[[16,4],[16,5],[16,6]]
wordCNN_config=[[64,3],[64,4],[64,5],[64,6]]
UNIT1=64
UNIT2=128
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9, epsilon=None, decay=0.001, amsgrad=True)
sgd =SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
losses = 'categorical_crossentropy'
class HGAN:
    def __init__(self,char_embedding,word_embedding,NUM_CLASS,dataset,language,task,CHAR_LEVEL=False,MAX_WORD=90,MAX_CHAR_WORD=25,lstm_unit=300,config_charCNN=CharCNN_config,config_wordCNN=wordCNN_config):
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
            e=np.random.normal(-1,1,embedding.shape[1])
            new_vec=np.multiply(vec,e)
            n_embedding.append(list(new_vec))
            #print(np.linalg.norm(vec),np.linalg.norm(new_vec))
        return np.asarray(n_embedding)
    def add_embedding(self,batch):
        batch_ret=[]
        for line in batch:
            line_ret=[]
            for index in line:
                line_ret.append(self.word_embedding[index])
            batch_ret.append(line_ret)
        return np.asarray(batch_ret)
    def build_model(self):
         #discriminator
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
            feature_out=Dropout(0.5)(feature_out)
            intent_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='intent_out')(feature_out)
            valid_out=Dense(units=2, activation='sigmoid', kernel_initializer='he_normal',name ='valid_out')(feature_out)
            self.discriminator=Model(inputs=word_input, outputs=[valid_out,intent_out])  
        elif self.task=='slot':
            slot_in=Reshape((self.MAX_WORD,256))(concat_in)
            slot_out = Dense(units=UNIT2, activation='relu', kernel_initializer='he_normal')(slot_in)
            slot_out=Dropout(0.5)(slot_out)
            feature_out=Flatten()(concat_in)
            valid_out=Dense(units=2, activation='sigmoid', kernel_initializer='he_normal',name ='valid_out')(feature_out)
            slot_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='slot_out')(slot_out)
            self.discriminator=Model(inputs=word_input, outputs=[valid_out,slot_out])  
            
        self.discriminator.compile(loss=losses,optimizer=adam,metrics=['accuracy'])   
            
        # genrator    
        word_input_gen = Input(shape=(self.MAX_WORD,), dtype='float32',name='words_input_gen')
        embed_out=Embedding(self.n_embedding.shape[0], self.n_embedding.shape[1], weights=[self.n_embedding],trainable=False)(word_input_gen)
        lstm_out=LSTM(128, activation='relu',  kernel_initializer='random_uniform',return_sequences=True)(embed_out)
        bt=BatchNormalization()(lstm_out)
        lstm_out=LSTM(self.lstm_unit, activation='relu',  kernel_initializer='random_uniform',return_sequences=True)(bt)
        bt=BatchNormalization()(lstm_out)
        self.generator=Model(inputs=word_input_gen, outputs=bt)
        fake_sent=self.generator(word_input_gen)
        self.discriminator.trainable = True
        valid, target_label = self.discriminator(fake_sent)
        self.combined = Model(word_input_gen, [valid, target_label])
        self.combined.compile(loss=losses,optimizer=adam)
        self.combined.summary()
        self.discriminator.summary()
        self.generator.summary()
        return self.combined
    def train_model(self,graph,X_word,X_char,Y,X_word_valid,X_char_valid,Y_valid,epochs=500,batch_size=128):
        train_loss=[]
        train_accuracy=[]
        test_loss=[]
        test_accuracy=[]
        max_accuracy=0.0
        for ep in tqdm(range(epochs)):
            kf=ModelSupport.get_minibatches_id('',X_word.shape[0],batch_size)
            kf_test=ModelSupport.get_minibatches_id('',X_word_valid.shape[0],batch_size)
            batch_loss=0.0
            batch_accuracy=0.0
            batch_test_loss=0.0
            batch_test_accuracy=0.0
            batch_test_count=0
            batch_count=0
            for bt,idx in kf:
                X_train_word=X_word[idx]
                X_train_embed=self.add_embedding(X_train_word)
                Y_train=Y[idx]
                #discriminator training 
                true_Y=to_categorical(np.ones(len(idx),dtype='int32'))
                fake_Y=to_categorical(np.zeros(len(idx),dtype='int32'),2)
                fake_sent=self.generator.predict(X_train_word)
                
                
                a,loss,c,d,accuracy=self.discriminator.train_on_batch(X_train_embed,[true_Y,Y_train])
                a,fake_loss,c,d,fake_accuracy=self.discriminator.train_on_batch(fake_sent,[fake_Y,Y_train])
                
                batch_loss=batch_loss+loss
                batch_accuracy=batch_accuracy+accuracy
                batch_count=batch_count+1
                if(batch_count%1==0):  # change to see effect future
                    self.discriminator.trainable=False
                    _=graph.train_on_batch(X_train_word,[fake_Y,Y_train])
                    self.discriminator.trainable=True
                
                
                #print(batch_count)
            batch_loss=batch_loss/batch_count
            batch_accuracy=batch_accuracy/batch_count
            
            for bbt,idx in kf_test:
                X_test_word=X_word_valid[idx]
                X_test_embed=self.add_embedding(X_test_word)
                Y_test=Y_valid[idx]
                true_Y=to_categorical(np.ones(len(idx),dtype='int32'))
                a,test_loss_a,c,d,test_accuracy_a=self.discriminator.test_on_batch(X_test_embed,[true_Y,Y_test])
                batch_test_count=batch_test_count+1
                batch_test_loss=batch_test_loss+test_loss_a
                batch_test_accuracy=batch_test_accuracy+test_accuracy_a
            batch_test_loss=batch_test_loss/batch_test_count
            batch_test_accuracy=batch_test_accuracy/batch_test_count
            print('\nTraining loss ',batch_loss)
            print('Training Accuracy ',batch_accuracy)
            print('Test loss ',batch_test_loss)
            print('Test Accuracy ',batch_test_accuracy)
            
            if(batch_test_accuracy>max_accuracy):
                max_accuracy=batch_test_accuracy
                self.save_model(graph)
            train_loss.append(batch_loss)
            train_accuracy.append(batch_accuracy)
            test_loss.append(batch_test_loss)
            test_accuracy.append(batch_test_accuracy)
        return train_loss,train_accuracy,test_loss,test_accuracy,max_accuracy
    def save_model(self,model):
         model_path = './models/single_language/single_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HGRU.json'
         weights_path = './models/single_language/single_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HGRU_weights.hdf5'
         options = {"file_arch": model_path,"file_weight": weights_path}
         json_string = model.to_json()
         open(options['file_arch'], 'w').write(json_string)
         model.save_weights(options['file_weight'])