import pandas as pd
import json, re, uvicorn
from pydantic import BaseModel
from typing import List
from transformers import pipeline
from string import punctuation
import unicodedata2 as unicodedata
from fastapi import FastAPI, Query, HTTPException, Path

def lowercase(text):
    return text.lower()
def casefolding(s):
    new_str = s.lower()  
    return new_str

def masking_entity(str):
    new_url =  re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',"", str)
    return new_url

def cleaning(str):
    #remove digit from string
    str = re.sub("\S*\d\S*", "", str).strip()
    #removeHashtag
    str = re.sub('#[^\s]+','',str)
    #remove mention
    str = re.sub("@([a-zA-Z0-9_]{1,50})","",str)
    #remove non-ascii
    str = unicodedata.normalize('NFKD', str).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #remove_punctuation
    #str = str.translate(str.maketrans("", "", punctuation))
    #to lowercase
    str = str.lower()
    #Remove additional white spaces
    str = re.sub('[\s]+', ' ', str)
    
    return str

#slang word
def normalize_slang_word(str):
    text_list = str.split(' ')
    slang_words_raw = pd.read_csv('/setneg-dir-02/saas-socmed/dev-nlp/dev-divertme/slang_word_list.csv', sep=',', header=None)
    slang_word_dict = {}
    
    for item in slang_words_raw.values:
        slang_word_dict[item[0]] = item[1]
        
        for index in range(len(text_list)):
            if text_list[index] in slang_word_dict.keys():
                text_list[index] = slang_word_dict[text_list[index]]
    
    return ' '.join(text_list)

def preprocess_text(text):
    text = casefolding(text)
    text = cleaning (text)
    text = masking_entity (text)
    text = "".join(text)
    text = normalize_slang_word(text)
    return text

path_model = '/setneg-dir/Uli/'

classifier = pipeline('sentiment-analysis', model = path_model+'best_model')

def prediction(text):
    result = []
    for i in text:
        prep = preprocess_text(i)
        predict = classifier(prep)
        for j in predict:
            if j['score']<=0.80:
                result.append(1)
            else:
                result.append(0)
    return text, result

def predict_user (prd_list):
    check = sum(prd_list)/len(prd_list)
    if check > 0.5:
        label = 'depression'
    else:
        label = 'control'
    return label

app = FastAPI()

class depression(BaseModel):
    user_id: str
    text: List[str]
    

@app.post("/predict_depression", status_code = 200)
async def final_pred(Items : depression):
    cek = prediction(Items.text)
    test = predict_user(cek[1])
    return {'user_id':Items.user_id, 'text':cek[0], 'prediction each sentence':cek[1], 'prediction for user '+Items.user_id :test}

if __name__ == '__main__':
    uvicorn.run("predict_depression:app")