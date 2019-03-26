#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:24:11 2019

@author: student
"""

from support.modelsupport import ModelSupport
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, concatenate,BatchNormalization,Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, add, TimeDistributed, Bidirectional, Lambda,Reshape
from evaluate.modelevaluate import ModelEvaluate
from tqdm import tqdm
from keras.optimizers import Adam
CharCNN_config=[[16,3],[16,4],[16,5]]
wordCNN_config=[[128,2],[128,3],[128,4],[128,5]]
UNIT1=512
UNIT2=128
optimizer = Adam(lr=0.0001, beta_1=0.90, beta_2=0.9, epsilon=None, decay=0, amsgrad=True)
losses = 'categorical_crossentropy'
class HCNN:
    def __init__(self,char_embedding,word_embedding,NUM_CLASS,dataset,language,task,CHAR_LEVEL=False,MAX_WORD=90,MAX_CHAR_WORD=25,config_charCNN=CharCNN_config,config_wordCNN=wordCNN_config):
        self.char_embedding=char_embedding
        self.word_embedding=word_embedding
        self.NUM_CLASS=NUM_CLASS
        self.CHAR_LEVEL=CHAR_LEVEL
        self.MAX_WORD=MAX_WORD
        self.MAX_CHAR_WORD=MAX_CHAR_WORD
        self.config_charCNN=CharCNN_config
        self.config_wordCNN=wordCNN_config
        self.dataset=dataset
        self.language=language
        self.task=task
        self.model='HCNN'
    def f1(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())   
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
            intent_out=Dropout(0.5)(intent_out)
            intent_out = Dense(units=self.NUM_CLASS, activation='softmax', kernel_initializer='he_normal',name ='intent_out')(intent_out)
            if self.CHAR_LEVEL:
                graph = Model(inputs=[char_input,word_input], outputs=intent_out)
            else:
                graph = Model(inputs=word_input, outputs=intent_out)
            graph.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])
            graph.summary()
            return graph
        elif self.task=='slot':
            slot_in=Reshape((self.MAX_WORD,256))(concat_in)
            slot_out = Dense(units=UNIT1, activation='relu', kernel_initializer='he_normal')(slot_in)
            slot_out=Dropout(0.1)(slot_out)
            slot_out = Dense(units=self.NUM_CLASS, activation='sigmoid', kernel_initializer='he_normal',name ='slot_out')(slot_out)
            if self.CHAR_LEVEL:
                graph = Model(inputs=[char_input,word_input], outputs=slot_out)
            else:
                graph = Model(inputs=word_input, outputs=slot_out)
            graph.compile(loss=losses,optimizer=optimizer,metrics=[self.f1])
            graph.summary()
            return graph
            
            
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
         model_path = './models/single_language/single_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HCNN.json'
         weights_path = './models/single_language/single_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HCNN_weights.hdf5'
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
            
        
        

        