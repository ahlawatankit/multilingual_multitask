# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:45:20 2019

@author: HP
"""
from support.embeddingdictionary import EmbeddingDictionary
import numpy as np
import os
class slot:
     def read_slot(self,dataset_name,lang):
        slots= EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset_name+'/'+lang+'/slot_train.txt'))
        slots.extend(EmbeddingDictionary.read_file(self,os.path.abspath('./datasets/'+dataset_name+'/'+lang+'/slot_test.txt')))
        return slots
     def build_dict(self,slots):
        slot2id={}
        id2slot={}
        slot2id['<PAD>']=0
        id2slot[0]='<PAD>'
        index=1
        for slot_line in slots:
            line=slot_line.split(' ')
            for label in line:
                if label not in slot2id:
                    slot2id[label]=index
                    id2slot[index]=label
                    index=index+1
        return slot2id,id2slot
     def encode_slot(self,slots,slot2id,max_length):
        en_slot=[]
        for slot_line in slots:
            line=slot_line.split(' ')
            line_slot=[]
            count=max_length
            for label in line:
                line_slot.append(slot2id[label])
                count=count-1
            while(count>0):
                line_slot.append(slot2id['<PAD>'])
                count=count-1
            en_slot.append(line_slot)
        return np.asarray(en_slot,dtype='int32')
     def decode_slot(self,en_slots,id2slot):
        slots=[]
        for slot_line in en_slots:
            line = slot_line.split(' ')
            line_slot=[]
            for label in line:
                line_slot.append(id2slot[label])
            slots.append(line_slot)
        return slots