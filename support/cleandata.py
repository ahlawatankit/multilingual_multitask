#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:55:41 2019

@author: student
"""
import datetime
class CleanData:
    @staticmethod
    def check_date(word):
        #date formate mm/dd/yy
        date=word.split('/')
        if(len(date)<2 or len(date)>3):
            return word
        if(len(date)==2):
            try :
                datetime.datetime(2015,int(date[0]),int(date[1]))
            except ValueError :
                return word
        if(len(date)==3):
             try :
                datetime.datetime(int(date[2]),int(date[0]),int(date[1]))
             except ValueError :
                return word
        return '#date'
    @staticmethod
    def isFloat(word):
        try:
            float(word)
            return '#num'
        except ValueError:
            return word
    @staticmethod
    def isInt(word):
        try:
            int(word)
            return '#num'
        except ValueError:
            return word
    @staticmethod
    def isTime(word):
        time=word.split(':')
        if(len(time)==2):
            if(CleanData.isInt(time[0]) and CleanData.isInt(time[1])):
                return '#time'
        return word
    
    def adding_tokens(self,sentence):
        sent=[]
        for line in sentence:
            line1=[]
            for word in line.split():
                word=word.lower()
                word=CleanData.isInt(word)
                word=CleanData.isFloat(word)
                word=CleanData.isTime(word)
                word=CleanData.check_date(word)
                line1.append(word)
            line1=' '.join(line1)
            sent.append(line1)
        return sent