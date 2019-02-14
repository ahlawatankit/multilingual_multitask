import pandas as pd
import os

intent_train=pd.read_csv(os.path.abspath('./datasets/ATIS/english/intent_train.txt'),header=None)
intent_train=list(intent_train[0])
hindi_data=pd.read_csv(os.path.abspath('./datasets/ATIS/hindi/slot_train.csv'),header=None)
hindi_sent=pd.read_csv(os.path.abspath('utterence_train.txt'),header=None)

hindi_sent=list(hindi_sent[0])
hindi_slot=list(hindi_data[2])

problem=list(hindi_data[3])

count=0
fp1=open('train_slot.txt','w')
fp2=open('intent_train.txt','w')
fp3=open('utterence_train.txt','wb')
for idx in range(len(problem)):
    if str(problem[idx])=='nan':
        fp1.write(hindi_slot[idx]+'\n')
        fp2.write(intent_train[idx]+'\n')
        fp3.write((hindi_sent[idx]+'\n').encode('utf-8'))
        #fp3.write(b'\n')
        count=count+1
fp1.close()
fp2.close()
fp3.close()

fp=open('utterence_train.txt','r',encoding='utf-8')
for line in fp:
    print(line)
fp.close()