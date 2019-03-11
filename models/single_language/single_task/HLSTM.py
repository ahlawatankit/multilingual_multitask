#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:24:11 2019

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
from keras.optimizers import Adam,SGD
CharCNN_config=[[16,3],[16,4]]
UNIT1=128
UNIT2=128
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.88, decay=0.0001, amsgrad=False)
sgd =SGD(lr=0.0001, decay=1e-5, momentum=0.95, nesterov=True)
losses = 'categorical_crossentropy'
class HLSTM:
    def __init__(self,char_embedding,word_embedding,NUM_CLASS,dataset,language,task,CHAR_LEVEL=False,MAX_WORD=90,MAX_CHAR_WORD=25,config_charCNN=CharCNN_config,lstm_hidden_size=128):
        self.char_embedding=char_embedding
        self.word_embedding=word_embedding
        self.NUM_CLASS=NUM_CLASS
        self.CHAR_LEVEL=CHAR_LEVEL
        self.MAX_WORD=MAX_WORD
        self.MAX_CHAR_WORD=MAX_CHAR_WORD
        self.config_charCNN=CharCNN_config
        self.lstm_hidden_size=lstm_hidden_size
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
        embed_word_out = Embedding(self.word_embedding.shape[0], self.word_embedding.shape[1], weights=[self.word_embedding],trainable=False, mask_zero = False)(word_input)
        if self.CHAR_LEVEL:
            embed_word_out=concatenate([char_out,embed_word_out],axis=-1)        
        if self.task=='intent':
            flat_in=LSTM(self.lstm_hidden_size, activation='relu', kernel_initializer='glorot_uniform',return_sequences=False)(embed_word_out)
            #flat_in=Dropout(0.1)(flat_in)
            intent_out = Dense(units=UNIT2, activation='relu', kernel_initializer='glorot_uniform')(flat_in)
            intent_out=Dropout(0.1)(intent_out)
            intent_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='glorot_uniform',name ='intent_out')(intent_out)
            if self.CHAR_LEVEL:
                graph = Model(inputs=[char_input,word_input], outputs=intent_out)
            else:
                graph = Model(inputs=word_input, outputs=intent_out)
            graph.compile(loss=losses,optimizer=adam,metrics=['accuracy'])
            graph.summary()
            return graph
        elif self.task=='slot':
            flat_in=LSTM(self.lstm_hidden_size, activation='relu', kernel_initializer='he_normal',return_sequences=True)(embed_word_out)
            slot_out = Dense(units=UNIT2, activation='relu', kernel_initializer='glorot_uniform')(flat_in)
            slot_out=Dropout(0.5)(slot_out)
            slot_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='glorot_uniform',name ='slot_out')(slot_out)
            if self.CHAR_LEVEL:
                graph = Model(inputs=[char_input,word_input], outputs=slot_out)
            else:
                graph = Model(inputs=word_input, outputs=slot_out)
            graph.compile(loss=losses,optimizer=adam,metrics=['accuracy'])
            graph.summary()
            return graph
            
            
    def train_model(self,graph,X_word,X_char,Y,X_word_valid,X_char_valid,Y_valid,epochs=500,batch_size=16):
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
                if self.CHAR_LEVEL:
                    X_train_char=X_char[idx]
                X_train_word=X_word[idx]
                Y_train=Y[idx]
                if self.CHAR_LEVEL:
                    loss,accuracy=graph.train_on_batch([X_train_char,X_train_word],Y_train)
                else:
                    loss,accuracy=graph.train_on_batch(X_train_word,Y_train)
                batch_loss=batch_loss+loss
                batch_accuracy=batch_accuracy+accuracy
                batch_count=batch_count+1
                #print(batch_count)
            batch_loss=batch_loss/batch_count
            batch_accuracy=batch_accuracy/batch_count
            
            for bbt,idx in kf_test:
                if self.CHAR_LEVEL:
                    X_test_char=X_char_valid[idx]
                X_test_word=X_word_valid[idx]
                Y_test=Y_valid[idx]
                if self.CHAR_LEVEL:
                    test_loss_a,test_accuracy_a=graph.test_on_batch([X_test_char,X_test_word],Y_test)
                else:
                    test_loss_a,test_accuracy_a=graph.test_on_batch(X_test_word,Y_test)
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
         model_path = './models/single_language/single_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HLSTM.json'
         weights_path = './models/single_language/single_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HLSTM_weights.hdf5'
         options = {"file_arch": model_path,"file_weight": weights_path}
         json_string = model.to_json()
         open(options['file_arch'], 'w').write(json_string)
         model.save_weights(options['file_weight'])
        
    def test_model(self,graph,X_word,X_char,Y):
        if self.CHAR_LEVEL:
            pred=graph.predict([X_char,X_word])
            pred=np.argmax(pred,axis=1)
            if self.task=='intent':
                return ModelEvaluate.accuracy('',Y,pred)
            elif self.task=='slot':
                return ModelEvaluate.f1('',Y,pred)
        else:
            pred=graph.predict(X_word)
            pred=np.argmax(pred,axis=1)
            if self.task=='intent':
                return ModelEvaluate.accuracy('',Y,pred)
            elif self.task=='slot':
                return ModelEvaluate.f1('',Y,pred)
            
        
        

        