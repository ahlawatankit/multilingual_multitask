#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:45:31 2019

@author: student
"""
from support.embeddingdictionary import EmbeddingDictionary
from support.translatedataset import TranslateDataset
import os 
class preprocess:
    def read_all(self,language):
        if(language=='english'):
            english=TranslateDataset.read_file(self,os.path.abspath('./datasets/ATIS/english/utterence_train.txt'))
            english.extend(TranslateDataset.read_file(self,os.path.abspath('./datasets/ATIS/english/utterence_test.txt')))
            english.extend(TranslateDataset.read_file(self,os.path.abspath('./datasets/Frames_data/english/utterence_train.txt')))
            english.extend(TranslateDataset.read_file(self,os.path.abspath('./datasets/Frames_data/english/utterence_test.txt')))
            english.extend(TranslateDataset.read_file(self,os.path.abspath('./datasets/Trains_dataset/english/utterence_train.txt')))
            english.extend(TranslateDataset.read_file(self,os.path.abspath('./datasets/Trains_dataset/english/utterence_test.txt')))
            return english
    def dict_embed(self,sent):
         word2id,id2word=EmbeddingDictionary.create_dictionary(self,sent)
         return word2id,id2word
        
        
        
