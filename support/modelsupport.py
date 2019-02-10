# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 13:00:06 2019

@author: HP
"""

import numpy as np
class ModelSupport:
    def get_minibatches_id(self,n,batch_size,shuffle=False):
        idx=np.arange(n,dtype="int32")
        if shuffle:
            np.random.shuffle(idx)
        minibatch=[]
        minibatch_start=0
        for i in range(n // batch_size):
            minibatch.append(idx[minibatch_start:minibatch_start + batch_size])
            minibatch_start+=batch_size
        if(minibatch_start!=n):
            minibatch.append(idx[minibatch_start:])
        return zip(range(len(minibatch)),minibatch)
    