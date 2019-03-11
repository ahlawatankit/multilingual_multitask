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
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, add, TimeDistributed, Bidirectional, Lambda,Reshape,GRU
from evaluate.modelevaluate import ModelEvaluate
from tqdm import tqdm
from keras.optimizers import Adam
CharCNN_config=[[16,3],[16,4],[16,5]]
UNIT1=64
UNIT2=128
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.95, epsilon=None, decay=0.00001, amsgrad=True)
losses = 'categorical_crossentropy'
class HGRU:
    def __init__(self,char_embedding,word_embedding,INTENT_CLASS,SLOT_CLASS,dataset,language,task,CHAR_LEVEL=False,MAX_WORD=90,MAX_CHAR_WORD=25,config_charCNN=CharCNN_config,gru_hidden_size=128):
        self.char_embedding=char_embedding
        self.word_embedding=word_embedding
        self.INTENT_CLASS=INTENT_CLASS
        self.SLOT_CLASS=SLOT_CLASS
        self.CHAR_LEVEL=CHAR_LEVEL
        self.MAX_WORD=MAX_WORD
        self.MAX_CHAR_WORD=MAX_CHAR_WORD
        self.config_charCNN=CharCNN_config
        self.gru_hidden_size=gru_hidden_size
        self.dataset=dataset
        self.language=language
        self.task=task
        self.model='HGRU'
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
        seq_in=GRU(self.gru_hidden_size, activation='relu', kernel_initializer='he_normal',return_sequences=True)(embed_word_out)
            
        flat_in=Flatten()(seq_in)
        intent_out = Dense(units=UNIT2, activation='relu', kernel_initializer='he_normal')(flat_in)
        intent_out=Dropout(0.3)(intent_out)
        intent_out = Dense(units=self.INTENT_CLASS, activation='softmax', kernel_initializer='he_normal',name ='intent_out')(intent_out)
        slot_out = Dense(units=UNIT2, activation='relu', kernel_initializer='he_normal')(seq_in)
        slot_out=Dropout(0.3)(slot_out)
        slot_out = Dense(units=self.SLOT_CLASS, activation='softmax', kernel_initializer='he_normal',name ='slot_out')(slot_out)

        if self.CHAR_LEVEL:
            graph = Model(inputs=[char_input,word_input], outputs=[intent_out,slot_out])
        else:
            graph = Model(inputs=word_input, outputs=[intent_out,slot_out])
        graph.compile(loss=losses,optimizer=adam,metrics=['accuracy'])
        graph.summary()       
        return graph    
            
    def train_model(self,graph,X_word,X_char,Y_IN,Y_SO,X_word_valid,X_char_valid,Y_IN_valid,Y_SO_valid,epochs=500,batch_size=128):
        train_loss_IN=[]
        train_loss_SO=[]
        train_accuracy_IN=[]
        train_accuracy_SO=[]
        test_loss_IN=[]
        test_loss_SO=[]
        test_accuracy_IN=[]
        test_accuracy_SO=[]
        max_accuracy_IN=0.0
        max_accuracy_SO=0.0
        for ep in tqdm(range(epochs)):
            kf=ModelSupport.get_minibatches_id('',X_word.shape[0],batch_size)
            kf_test=ModelSupport.get_minibatches_id('',X_word_valid.shape[0],batch_size)
            batch_loss_IN=0.0
            batch_loss_SO=0.0
            batch_accuracy_IN=0.0
            batch_accuracy_SO=0.0
            batch_test_loss_IN=0.0
            batch_test_loss_SO=0.0
            batch_test_accuracy_IN=0.0
            batch_test_accuracy_SO=0.0
            batch_test_count=0
            batch_count=0
            for bt,idx in kf:
                if self.CHAR_LEVEL:
                    X_train_char=X_char[idx]
                X_train_word=X_word[idx]
                Y_IN_train=Y_IN[idx]
                Y_SO_train=Y_SO[idx]
                if self.CHAR_LEVEL:
                    loss_IN,loss_SO,combine_loss,accuracy_IN,accuracy_SO=graph.train_on_batch([X_train_char,X_train_word],[Y_IN_train,Y_SO_train])
                else:
                    loss_IN,loss_SO,cobine_loss,accuracy_IN,accuracy_SO=graph.train_on_batch(X_train_word,[Y_IN_train,Y_SO_train])
                batch_loss_IN=batch_loss_IN+loss_IN
                batch_loss_SO=batch_loss_SO+loss_SO
                batch_accuracy_IN=batch_accuracy_IN+accuracy_IN
                batch_accuracy_SO=batch_accuracy_SO+accuracy_SO
                batch_count=batch_count+1
                #print(batch_count)
            batch_loss_IN=batch_loss_IN/batch_count
            batch_loss_SO=batch_loss_SO/batch_count
            batch_accuracy_IN=batch_accuracy_IN/batch_count
            batch_accuracy_SO=batch_accuracy_SO/batch_count
            for bbt,idx in kf_test:
                if self.CHAR_LEVEL:
                    X_test_char=X_char_valid[idx]
                X_test_word=X_word_valid[idx]
                Y_IN_test=Y_IN_valid[idx]
                Y_SO_test=Y_SO_valid[idx]
                if self.CHAR_LEVEL:
                    test_loss_IN_a,test_loss_SO_a,combine_loss,test_accuracy_IN_a,test_accuracy_SO_a=graph.test_on_batch([X_test_char,X_test_word],[Y_IN_test,Y_SO_test])
                else:
                    test_loss_IN_a,test_loss_SO_a,combine_loss_a,test_accuracy_IN_a,test_accuracy_SO_a=graph.test_on_batch(X_test_word,[Y_IN_test,Y_SO_test])
                batch_test_count=batch_test_count+1
                batch_test_loss_IN=batch_test_loss_IN+test_loss_IN_a
                batch_test_loss_SO=batch_test_loss_SO+test_loss_SO_a
                batch_test_accuracy_IN=batch_test_accuracy_IN+test_accuracy_IN_a
                batch_test_accuracy_SO=batch_test_accuracy_SO+test_accuracy_SO_a
            batch_test_loss_IN=batch_test_loss_IN/batch_test_count
            batch_test_loss_SO=batch_test_loss_SO/batch_test_count
            batch_test_accuracy_IN=batch_test_accuracy_IN/batch_test_count
            batch_test_accuracy_SO=batch_test_accuracy_SO/batch_test_count
            print('\nIntent Training loss ',batch_loss_IN)
            print('Slot Training loss ',batch_loss_SO)
            print('Intent Test loss ',batch_test_loss_IN)
            print('Slot Test loss ',batch_test_loss_SO)
            print('Intent Training Accuracy ',batch_accuracy_IN)
            print('Slot Training Accuracy ',batch_accuracy_SO)           
            print('Intent Test Accuracy ',batch_test_accuracy_IN)
            print('Slot Test Accuracy ',batch_test_accuracy_SO)
            if(batch_test_accuracy_IN>max_accuracy_IN or batch_test_accuracy_SO>max_accuracy_SO):
                max_accuracy_IN=max(batch_test_accuracy_IN,max_accuracy_IN)
                max_accuracy_SO=max(batch_test_accuracy_SO,max_accuracy_SO)
                self.save_model(graph)
            train_loss_IN.append(batch_loss_IN)
            train_loss_SO.append(batch_loss_SO)
            train_accuracy_IN.append(batch_accuracy_IN)
            train_accuracy_SO.append(batch_accuracy_SO)
            test_loss_IN.append(batch_test_loss_IN)
            test_loss_SO.append(batch_test_loss_SO)
            test_accuracy_IN.append(batch_test_accuracy_IN)
            test_accuracy_SO.append(batch_test_accuracy_SO)
        return train_loss_IN,train_accuracy_IN,train_loss_SO,train_accuracy_SO,test_loss_IN,test_accuracy_IN,test_loss_SO,test_accuracy_SO,max_accuracy_IN,max_accuracy_SO
    def save_model(self,model):
         model_path = './models/single_language/multi_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HGRU.json'
         weights_path = './models/single_language/multi_task/'+self.dataset+'_'+self.language+'_'+self.task+'_'+'HGRU_weights.hdf5'
         options = {"file_arch": model_path,"file_weight": weights_path}
         json_string = model.to_json()
         open(options['file_arch'], 'w').write(json_string)
         model.save_weights(options['file_weight'])
        
    def test_model(self,graph,X_word,X_char,Y_IN,Y_SO):
        if self.CHAR_LEVEL:
            pred=graph.predict([X_char,X_word])
            pred_IN=np.argmax(pred[0],axis=-1)
            pred_SO=np.argmax(pred[1],axis=-1)
            return ModelEvaluate.accuracy('',Y_IN,pred_IN),ModelEvaluate.f1('',Y_SO,pred_SO)
        else:
            pred=graph.predict(X_word)
            pred_IN=np.argmax(pred[0],axis=-1)
            pred_SO=np.argmax(pred[1],axis=-1)
            return ModelEvaluate.accuracy('',Y_IN,pred_IN),ModelEvaluate.f1('',Y_SO,pred_SO)
            
        