#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:23:08 2019

@author: student
"""
from sklearn.metrics import f1_score
class ModelEvaluate:
    def accuracy(self,list1, list2):
        if(len(list1) != len(list2)):
            print("Size of a the lists not equal")
            return
        count = 0.0
        for i in range(len(list1)):
            if(list1[i] == list2[i]):
                count = count + 1                
        return count/len(list1)
    def flatten(self,mat):
        arr = list()
        for ar in mat:
            for a in ar:
                arr.append(a)
        return arr
    def f1(self,pred,actual):
        return f1_score(self.flatten(actual),self.flatten(pred), average = 'weighted')
        