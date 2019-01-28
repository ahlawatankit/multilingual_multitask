from google.cloud import translate
import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="intent.json"
#translate_client = translate.Client()
class TranslateDataset:
    def __init__(self,dataset_url,dataset_out_url):
        self.dataset_url=dataset_url
        self.dataset_out_url=dataset_out_url
    def read_file(self,file_name):
        ret_list=[]
        fp=open(file_name,'r')
        for line in fp:
            line=line.replace('\n','')
            ret_list.append(line)
        return ret_list
    def translate(self):
        sentences=self.read_file(self.dataset_url)
        with open(self.dataset_out_url,'w') as file:
            for sen in sentence:
                translation = translate_client.translate(sen,target_language=target)
                sen_hi=translation['translatedText']
                file.write(sen_hi+'\n')
                print(sen)
                print(sen_hi)
