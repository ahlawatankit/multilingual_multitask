from google.cloud import translate
import os
import pickle
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="intent.json"
translate_client = translate.Client()
class SlotArrange:
    def __init__(self,english_dataset,hindi_dataset,english_slot):
        self.english_dataset=english_dataset
        self.hindi_dataset=hindi_dataset
        self.english_slot=english_slot
    def read_dict(self):
         with open('hindi_to_english.pkl', 'rb') as f:
            return pickle.load(f)
    def list_english(self,hindi,hindi_to_english):
        english_list=[]
        for word in hindi:
            if(word not in hindi_to_english):
                target = 'en'
                translation = translate_client.translate(word,target_language=target)
                english_list.append(translation['translatedText'].lower())
                hindi_to_english[word]=translation['translatedText'].lower()
            else:
                english_list.append(hindi_to_english[word])
                #print(hindi_to_english[word])
        return english_list
    def read_file(self,file_name):
        ret_list=[]
        fp=open(file_name,'r')
        for line in fp:
            line=line.replace('\n','')
            ret_list.append(line)
        return ret_list
    def arrange(self):
        print(self.english_dataset)
        english=self.read_file(self.english_dataset)
        hindi=self.read_file(self.hindi_dataset)
        slot=self.read_file(self.english_slot)
        dict_hindi=self.read_dict()
        with open('slot_test.csv','w') as file:
            for sen_index in range(len(english)):
                slot_english=slot[sen_index].split()
                english_sen=english[sen_index].split()
                hindi_sen=hindi[sen_index].split()
                slot_hindi=['O' for _ in range(len(hindi_sen))]
                translate_hindi=self.list_english(hindi_sen,dict_hindi)
                #print(translate_hindi)
                problem=[]
                for slot_index in range(len(slot_english)):
                    if(slot_english[slot_index]!='O'):
                        english_word=english_sen[slot_index]
                        try:
                            index_hi=translate_hindi.index(english_word)
                            slot_hindi[index_hi]=slot_english[slot_index]
                        except:
                            problem.append(english_word)
                file.write(english[sen_index].replace(',',' ')+','+hindi[sen_index].replace(',',' ')+','+" ".join(slot_hindi)+','+' '.join(problem)+'\n')
                #print(hindi[sen_index])
                print(sen_index)
