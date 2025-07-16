import numpy as np 
import pandas as pd 
import pickle 
import tensorflow as tf 
import streamlit as st
from tensorflow.keras.models import load_model

model=load_model('model.h5',compile=False)
with open('label.pkl','rb') as f:
    label=pickle.load(f)
with open('one.pkl','rb') as f:
    onehot=pickle.load(f)
with open('standed.pkl','rb') as f:
    stander=pickle.load(f)


st.title('customer salary')
CreditScore=st.number_input('cerdictscore')
Geography=st.selectbox('Geography',onehot.categories_[0])
Gender=st.selectbox('Gender',label.classes_)
Age=st.slider('Age',18,92)
Tenure=st.slider('Tenure',1,10)
Balance=st.number_input('Balance')
NumOfProducts=st.slider('NumOfProducts',1,4)
HasCrCard=st.selectbox('HasCrCard',[0,1])
IsActiveMember=st.selectbox('IsActiveMember',[0,1])
Exited=st.selectbox('Exited',[0,1])

input_data=pd.DataFrame({
    'CreditScore':[CreditScore],
    'Gender':[label.transform([Gender])[0]],
    'Age':[Age],
    'Tenure':[Tenure],
    'Balance':[Balance],
    'NumOfProducts':[NumOfProducts],
    'HasCrCard':[HasCrCard],
    'IsActiveMember':[IsActiveMember],
    'Exited':[Exited]

})
onehot_start=onehot.transform([[Geography]]).toarray()
df_obect=onehot.get_feature_names_out(['Geography'])
df_dataframe=pd.DataFrame(onehot_start,columns=df_obect)
input_data=pd.concat([input_data.reset_index(drop=True),df_dataframe],axis=1)
stand_scaler=stander.transform(input_data)
model_prediction=model.predict(stand_scaler)
if st.button('predict salary'):
   st.write(model_prediction)


