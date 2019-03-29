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
from keras.layers import Dense, Input, Flatten, Dropout, concatenate,BatchNormalization,Activation,GRU
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, add, TimeDistributed, Bidirectional, Lambda,Reshape
from evaluate.modelevaluate import ModelEvaluate
from tqdm import tqdm
from keras.optimizers import Adam,SGD
CharCNN_config=[[16,3],[16,4],[16,5]]
wordCNN_config=[[256,2],[256,3],[256,4],[256,5]]
UNIT1=64
UNIT2=129
adam = Adam(lr=0.0001, beta_1=0.90, beta_2=0.95, epsilon=None, decay=0.0000001, amsgrad=True)
sgd =SGD(lr=0.00001, decay=1e-6, momentum=0.95, nesterov=True)
losses = 'categorical_crossentropy'
class HGRU:
    def __init__(self,char_embedding,word_embedding,INTENT_CLASS1,SLOT_CLASS1,INTENT_CLASS2,SLOT_CLASS2,dataset,language1,language2,task,CHAR_LEVEL=False,MAX_WORD=90,MAX_CHAR_WORD=25,config_charCNN=CharCNN_config,config_wordCNN=wordCNN_config,gru_hidden_size=128):
        self.char_embedding=char_embedding
        self.word_embedding=word_embedding
        self.INTENT_CLASS1=INTENT_CLASS1
        self.SLOT_CLASS1=SLOT_CLASS1
        self.INTENT_CLASS2=INTENT_CLASS2
        self.SLOT_CLASS2=SLOT_CLASS2
        self.CHAR_LEVEL=CHAR_LEVEL
        self.MAX_WORD=MAX_WORD
        self.MAX_CHAR_WORD=MAX_CHAR_WORD
        self.config_charCNN=CharCNN_config
        self.config_wordCNN=wordCNN_config
        self.dataset=dataset
        self.language1=language1
        self.language2=language2
        self.task=task
        self.gru_hidden_size=gru_hidden_size
        self.model='HGRU'
    def build_model(self):
        #character CNN implemention
        if self.CHAR_LEVEL:
            char_input=Input(shape=(self.MAX_WORD,self.MAX_CHAR_WORD,),dtype='float32',name='Chars_input')
            embed_out=TimeDistributed(Embedding(self.char_embedding.shape[0],self.char_embedding.shape[1],weights=[self.char_embedding],trainable=False))(char_input)
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
        embed_word_out = Embedding(self.word_embedding.shape[0], self.word_embedding.shape[1], weights=[self.word_embedding],trainable=True)(word_input)
        if self.CHAR_LEVEL:
            embed_word_out=concatenate([char_out,embed_word_out],axis=-1)
        
        concat_in=GRU(self.gru_hidden_size, activation='relu', kernel_initializer='he_normal',return_sequences=True)(embed_word_out)
        concat_in=BatchNormalization()(concat_in)
        concat_in=Dropout(0.1)(concat_in)
        if self.CHAR_LEVEL:
            half_model=Model(inputs=[char_input,word_input], outputs=concat_in)
            intermediate_feature=half_model([char_input,word_input])
        else:
            half_model=Model(inputs=word_input, outputs=concat_in)
            intermediate_feature=half_model(word_input)
        
            
       #first language model
        flat_in=Flatten()(intermediate_feature)
        intent_out_lang1 = Dense(units=UNIT1, activation='relu', kernel_initializer='random_uniform')(flat_in)
        intent_out_lang1=Dropout(0.2)(intent_out_lang1)
        intent_out_lang1 = Dense(units=self.INTENT_CLASS1, activation='softmax', kernel_initializer='random_uniform',name ='intent_out_lang1')(intent_out_lang1)
        slot_in_lang1=intermediate_feature
        slot_in_lang1 = Dense(units=UNIT2, activation='relu', kernel_initializer='random_uniform')(slot_in_lang1)
       # slot_out=Dropout(0.1)(slot_out)
        slot_in_lang1 = Dense(units=self.SLOT_CLASS1, activation='softmax', kernel_initializer='random_uniform',name ='slot_in_lang1')(slot_in_lang1)
        if self.CHAR_LEVEL:
            language_1_model = Model(inputs=[char_input,word_input], outputs=[intent_out_lang1,slot_in_lang1])
        else:
            language_1_model = Model(inputs=word_input, outputs=[intent_out_lang1,slot_in_lang1])
        mt={'intent_out_lang1':'accuracy','slot_in_lang1':'acc'}
        language_1_model.compile(loss=losses,optimizer=adam,metrics=mt)
        language_1_model.summary()  
        
        #second language _model
        flat_in=Flatten()(intermediate_feature)
        intent_out_lang2 = Dense(units=UNIT1, activation='relu', kernel_initializer='random_uniform')(flat_in)
        intent_out_lang2=Dropout(0.2)(intent_out_lang2)
        intent_out_lang2 = Dense(units=self.INTENT_CLASS2, activation='softmax', kernel_initializer='random_uniform',name ='intent_out_lang2')(intent_out_lang2)
        slot_in_lang2=intermediate_feature
        slot_in_lang2 = Dense(units=UNIT2, activation='relu', kernel_initializer='random_uniform')(slot_in_lang2)
       # slot_out=Dropout(0.1)(slot_out)
        slot_in_lang2 = Dense(units=self.SLOT_CLASS2, activation='softmax', kernel_initializer='random_uniform',name ='slot_in_lang2')(slot_in_lang2)
        if self.CHAR_LEVEL:
            language_2_model = Model(inputs=[char_input,word_input], outputs=[intent_out_lang2,slot_in_lang2])
        else:
            language_2_model = Model(inputs=word_input, outputs=[intent_out_lang2,slot_in_lang2])
        mt={'intent_out_lang2':'accuracy','slot_in_lang2':'acc'}
        language_2_model.compile(loss=losses,optimizer=adam,metrics=mt)
        language_2_model.summary()  
        
        
        
        return language_1_model,language_2_model
            
            
    def train_model(self,language_1_model,language_2_model,X_word_lang1,X_char_lang1,Y_IN_lang1,Y_SO_lang1,X_word_valid_lang1,X_char_valid_lang1,Y_IN_valid_lang1,Y_SO_valid_lang1,
                    X_word_lang2,X_char_lang2,Y_IN_lang2,Y_SO_lang2,X_word_valid_lang2,X_char_valid_lang2,Y_IN_valid_lang2,Y_SO_valid_lang2,epochs=500,batch_size=64):
        train_loss_IN_lang1=[]
        train_loss_SO_lang1=[]
        train_accuracy_IN_lang1=[]
        train_accuracy_SO_lang1=[]
        test_loss_IN_lang1=[]
        test_loss_SO_lang1=[]
        test_accuracy_IN_lang1=[]
        test_accuracy_SO_lang1=[]
        train_loss_IN_lang2=[]
        train_loss_SO_lang2=[]
        train_accuracy_IN_lang2=[]
        train_accuracy_SO_lang2=[]
        test_loss_IN_lang2=[]
        test_loss_SO_lang2=[]
        test_accuracy_IN_lang2=[]
        test_accuracy_SO_lang2=[]
        max_accuracy_IN_lang1=0.0
        max_accuracy_SO_lang1=0.0
        max_accuracy_IN_lang2=0.0
        max_accuracy_SO_lang2=0.0
        for ep in tqdm(range(epochs)):
            kf1=ModelSupport.get_minibatches_id('',X_word_lang1.shape[0],batch_size)
            kf1_test=ModelSupport.get_minibatches_id('',X_word_valid_lang1.shape[0],batch_size)
            
            kf2=ModelSupport.get_minibatches_id('',X_word_lang2.shape[0],batch_size)
            kf2_test=ModelSupport.get_minibatches_id('',X_word_valid_lang2.shape[0],batch_size)
            
            batch_loss_IN_lang1=0.0
            batch_loss_SO_lang1=0.0
            batch_accuracy_IN_lang1=0.0
            batch_accuracy_SO_lang1=0.0
            batch_test_loss_IN_lang1=0.0
            batch_test_loss_SO_lang1=0.0
            batch_test_accuracy_IN_lang1=0.0
            batch_test_accuracy_SO_lang1=0.0
            
            batch_loss_IN_lang2=0.0
            batch_loss_SO_lang2=0.0
            batch_accuracy_IN_lang2=0.0
            batch_accuracy_SO_lang2=0.0
            batch_test_loss_IN_lang2=0.0
            batch_test_loss_SO_lang2=0.0
            batch_test_accuracy_IN_lang2=0.0
            batch_test_accuracy_SO_lang2=0.0
            
            batch_test_count_lang1=0
            batch_count_lang1=0
            batch_test_count_lang2=0
            batch_count_lang2=0
            #language one training 
            for bt,idx in kf1:
                if self.CHAR_LEVEL:
                    X_train_char=X_char_lang1[idx]
                X_train_word=X_word_lang1[idx]
                Y_IN_train=Y_IN_lang1[idx]
                Y_SO_train=Y_SO_lang1[idx]
                if self.CHAR_LEVEL:
                    loss_IN,loss_SO,combine_loss,accuracy_IN,accuracy_SO=language_1_model.train_on_batch([X_train_char,X_train_word],[Y_IN_train,Y_SO_train])
                else:
                    loss_IN,loss_SO,cobine_loss,accuracy_IN,accuracy_SO=language_1_model.train_on_batch(X_train_word,[Y_IN_train,Y_SO_train])
                batch_loss_IN_lang1=batch_loss_IN_lang1+loss_IN
                batch_loss_SO_lang1=batch_loss_SO_lang1+loss_SO
                batch_accuracy_IN_lang1=batch_accuracy_IN_lang1+accuracy_IN
                batch_accuracy_SO_lang1=batch_accuracy_SO_lang1+accuracy_SO
                batch_count_lang1=batch_count_lang1+1
                #print(batch_count)
            batch_loss_IN_lang1=batch_loss_IN_lang1/batch_count_lang1
            batch_loss_SO_lang1=batch_loss_SO_lang1/batch_count_lang1
            batch_accuracy_IN_lang1=batch_accuracy_IN_lang1/batch_count_lang1
            batch_accuracy_SO_lang1=batch_accuracy_SO_lang1/batch_count_lang1
            
            #language 2 training 
            for bt,idx in kf2:
                if self.CHAR_LEVEL:
                    X_train_char=X_char_lang2[idx]
                X_train_word=X_word_lang2[idx]
                Y_IN_train=Y_IN_lang2[idx]
                Y_SO_train=Y_SO_lang2[idx]
                if self.CHAR_LEVEL:
                    loss_IN,loss_SO,combine_loss,accuracy_IN,accuracy_SO=language_2_model.train_on_batch([X_train_char,X_train_word],[Y_IN_train,Y_SO_train])
                else:
                    loss_IN,loss_SO,cobine_loss,accuracy_IN,accuracy_SO=language_2_model.train_on_batch(X_train_word,[Y_IN_train,Y_SO_train])
                batch_loss_IN_lang2=batch_loss_IN_lang2+loss_IN
                batch_loss_SO_lang2=batch_loss_SO_lang2+loss_SO
                batch_accuracy_IN_lang2=batch_accuracy_IN_lang2+accuracy_IN
                batch_accuracy_SO_lang2=batch_accuracy_SO_lang2+accuracy_SO
                batch_count_lang2=batch_count_lang2+1
                #print(batch_count)
            batch_loss_IN_lang2=batch_loss_IN_lang2/batch_count_lang2
            batch_loss_SO_lang2=batch_loss_SO_lang2/batch_count_lang2
            batch_accuracy_IN_lang2=batch_accuracy_IN_lang2/batch_count_lang2
            batch_accuracy_SO_lang2=batch_accuracy_SO_lang2/batch_count_lang2
            
            #language 1 test
            for bbt,idx in kf1_test:
                if self.CHAR_LEVEL:
                    X_test_char=X_char_valid_lang1[idx]
                X_test_word=X_word_valid_lang1[idx]
                Y_IN_test=Y_IN_valid_lang1[idx]
                Y_SO_test=Y_SO_valid_lang1[idx]
                if self.CHAR_LEVEL:
                    test_loss_IN_a,test_loss_SO_a,combine_loss,test_accuracy_IN_a,test_accuracy_SO_a=language_1_model.test_on_batch([X_test_char,X_test_word],[Y_IN_test,Y_SO_test])
                else:
                    test_loss_IN_a,test_loss_SO_a,combine_loss_a,test_accuracy_IN_a,test_accuracy_SO_a=language_1_model.test_on_batch(X_test_word,[Y_IN_test,Y_SO_test])
                batch_test_count_lang1=batch_test_count_lang1+1
                batch_test_loss_IN_lang1=batch_test_loss_IN_lang1+test_loss_IN_a
                batch_test_loss_SO_lang1=batch_test_loss_SO_lang1+test_loss_SO_a
                batch_test_accuracy_IN_lang1=batch_test_accuracy_IN_lang1+test_accuracy_IN_a
                batch_test_accuracy_SO_lang1=batch_test_accuracy_SO_lang1+test_accuracy_SO_a
            batch_test_loss_IN_lang1=batch_test_loss_IN_lang1/batch_test_count_lang1
            batch_test_loss_SO_lang1=batch_test_loss_SO_lang1/batch_test_count_lang1
            batch_test_accuracy_IN_lang1=batch_test_accuracy_IN_lang1/batch_test_count_lang1
            batch_test_accuracy_SO_lang1=batch_test_accuracy_SO_lang1/batch_test_count_lang1
            
            #language 2 test
            for bbt,idx in kf2_test:
                if self.CHAR_LEVEL:
                    X_test_char=X_char_valid_lang2[idx]
                X_test_word=X_word_valid_lang2[idx]
                Y_IN_test=Y_IN_valid_lang2[idx]
                Y_SO_test=Y_SO_valid_lang2[idx]
                if self.CHAR_LEVEL:
                    test_loss_IN_a,test_loss_SO_a,combine_loss,test_accuracy_IN_a,test_accuracy_SO_a=language_2_model.test_on_batch([X_test_char,X_test_word],[Y_IN_test,Y_SO_test])
                else:
                    test_loss_IN_a,test_loss_SO_a,combine_loss_a,test_accuracy_IN_a,test_accuracy_SO_a=language_2_model.test_on_batch(X_test_word,[Y_IN_test,Y_SO_test])
                batch_test_count_lang2=batch_test_count_lang2+1
                batch_test_loss_IN_lang2=batch_test_loss_IN_lang2+test_loss_IN_a
                batch_test_loss_SO_lang2=batch_test_loss_SO_lang2+test_loss_SO_a
                batch_test_accuracy_IN_lang2=batch_test_accuracy_IN_lang2+test_accuracy_IN_a
                batch_test_accuracy_SO_lang2=batch_test_accuracy_SO_lang2+test_accuracy_SO_a
            batch_test_loss_IN_lang2=batch_test_loss_IN_lang2/batch_test_count_lang2
            batch_test_loss_SO_lang2=batch_test_loss_SO_lang2/batch_test_count_lang2
            batch_test_accuracy_IN_lang2=batch_test_accuracy_IN_lang2/batch_test_count_lang2
            batch_test_accuracy_SO_lang2=batch_test_accuracy_SO_lang2/batch_test_count_lang2            
            
            
            print('\nLanguage  '+self.language1)
            print('\nIntent Training loss ',batch_loss_IN_lang1)
            print('Slot Training loss ',batch_loss_SO_lang1)
            print('Intent Test loss ',batch_test_loss_IN_lang1)
            print('Slot Test loss ',batch_test_loss_SO_lang1)
            print('Intent Training Accuracy ',batch_accuracy_IN_lang1)
            print('Slot Training Accuracy ',batch_accuracy_SO_lang1)           
            print('Intent Test Accuracy ',batch_test_accuracy_IN_lang1)
            print('Slot Test Accuracy ',batch_test_accuracy_SO_lang1)
            
            print('\nLanguage  '+self.language2)
            print('\nIntent Training loss ',batch_loss_IN_lang2)
            print('Slot Training loss ',batch_loss_SO_lang2)
            print('Intent Test loss ',batch_test_loss_IN_lang2)
            print('Slot Test loss ',batch_test_loss_SO_lang2)
            print('Intent Training Accuracy ',batch_accuracy_IN_lang2)
            print('Slot Training Accuracy ',batch_accuracy_SO_lang2)           
            print('Intent Test Accuracy ',batch_test_accuracy_IN_lang2)
            print('Slot Test Accuracy ',batch_test_accuracy_SO_lang2)
            
            max_accuracy_IN_lang1=max(batch_test_accuracy_IN_lang1,max_accuracy_IN_lang1)
            max_accuracy_SO_lang1=max(batch_test_accuracy_SO_lang1,max_accuracy_SO_lang1)
            max_accuracy_IN_lang2=max(batch_test_accuracy_IN_lang2,max_accuracy_IN_lang2)
            max_accuracy_SO_lang2=max(batch_test_accuracy_SO_lang2,max_accuracy_SO_lang2)
                #self.save_model()
            train_loss_IN_lang1.append(batch_loss_IN_lang1)
            train_loss_SO_lang1.append(batch_loss_SO_lang1)
            train_accuracy_IN_lang1.append(batch_accuracy_IN_lang1)
            train_accuracy_SO_lang1.append(batch_accuracy_SO_lang1)
            test_loss_IN_lang1.append(batch_test_loss_IN_lang1)
            test_loss_SO_lang1.append(batch_test_loss_SO_lang1)
            test_accuracy_IN_lang1.append(batch_test_accuracy_IN_lang1)
            test_accuracy_SO_lang1.append(batch_test_accuracy_SO_lang1)
            
            train_loss_IN_lang2.append(batch_loss_IN_lang2)
            train_loss_SO_lang2.append(batch_loss_SO_lang2)
            train_accuracy_IN_lang2.append(batch_accuracy_IN_lang2)
            train_accuracy_SO_lang2.append(batch_accuracy_SO_lang2)
            test_loss_IN_lang2.append(batch_test_loss_IN_lang2)
            test_loss_SO_lang2.append(batch_test_loss_SO_lang2)
            test_accuracy_IN_lang2.append(batch_test_accuracy_IN_lang2)
            test_accuracy_SO_lang2.append(batch_test_accuracy_SO_lang2)
        return train_loss_IN_lang1,train_accuracy_IN_lang1,train_loss_SO_lang1,train_accuracy_SO_lang1,test_loss_IN_lang1,test_accuracy_IN_lang1,test_loss_SO_lang1,test_accuracy_SO_lang1,max_accuracy_IN_lang1,max_accuracy_SO_lang1,train_loss_IN_lang2,train_accuracy_IN_lang2,train_loss_SO_lang2,train_accuracy_SO_lang2,test_loss_IN_lang2,test_accuracy_IN_lang2,test_loss_SO_lang2,test_accuracy_SO_lang2,max_accuracy_IN_lang2,max_accuracy_SO_lang2
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
            
        
        

        
