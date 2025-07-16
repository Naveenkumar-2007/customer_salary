
from fastapi import FastAPI
import numpy as np 
import pandas as pd 
import pickle 
import tensorflow as tf 
from tensorflow.keras.models import load_model

model=load_model('model.h5',compile=False)
with open('label.pkl','rb') as f:
    label=pickle.load(f)
with open('one.pkl','rb') as f:
    onehot=pickle.load(f)
with open('standed.pkl','rb') as f:
    stander=pickle.load(f)

from pydantic import BaseModel

app=FastAPI()
class model_select(BaseModel):
    CreditScore:float
    Geography:str
    Gender:str
    Age:int
    Tenure:int
    Balance:float
    NumOfProducts:int
    HasCrCard:int
    IsActiveMember:int
    Exited:int


@app.get('/')
def start():
    return {"meassage":"hey welcome to salary prediction"}

@app.post('/predict')
def strat_next(data:model_select):
    input_dict=data.dict()
    Genderlabel=label.transform([input_dict['Gender']])[0]
    Geographyonehot=onehot.transform([[input_dict['Geography']]]).toarray()

    feautre=[
        input_dict['CreditScore'],
        Genderlabel,
        input_dict['Age'],
        input_dict['Tenure'],
        input_dict['Balance'],
        input_dict['NumOfProducts'],
        input_dict['HasCrCard'],
        input_dict['IsActiveMember'],
        input_dict['Exited']
    ]

    feautre=np.array([feautre]).reshape(1,-1)
    final=np.concatenate([feautre,Geographyonehot],axis=1)
    sta=stander.transform(final)
    model_pred=model.predict(sta)[0][0]
    return {"salary of customer:",float(model_pred)}

