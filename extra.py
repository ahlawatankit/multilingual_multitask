import pandas as pd
import os

intent_train=pd.read_csv(os.path.abspath('./datasets/Trains_dataset/english/intent_train.txt'),header=None)
intent_train=list(intent_train[0])
hindi_data=pd.read_csv(os.path.abspath('./datasets/Trains_dataset/ben/slot_train.csv'),header=None)

hindi_sent=list(hindi_data[1])
hindi_slot=list(hindi_data[2])

problem=list(hindi_data[3])

count=0
fp1=open('slot_train.txt','w')
fp2=open('intent_train.txt','w')
fp3=open('utterence_train.txt','w')
for idx in range(len(problem)):
    if str(hindi_sent[idx])!='nan':
        if (str(problem[idx])=='nan') or ((str(problem[idx])!='nan') and len(problem[idx].split(' '))<=1):
            #print(hindi_slot[idx])
            fp1.write(hindi_slot[idx]+'\n')
            fp2.write(intent_train[idx]+'\n')
            fp3.write(hindi_sent[idx]+'\n')
            #p3.write(b'\n')
            count=count+1
fp1.close()
fp2.close()
fp3.close()
