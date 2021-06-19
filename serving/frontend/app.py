"""
You need to run the app from the root
To run the app
$ streamlit run serving/frontend/app.py
"""

import numpy as np
import json
import joblib
import pandas as pd
import streamlit as st
from PIL import Image
import requests
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



with st.sidebar:
    st.subheader('Instruction!!!!')
    
    st.write("[https://github.com/tiennguyenhust/ais-dsp-tien](https://github.com/tiennguyenhust/ais-dsp-tien)")
    image = Image.open('data/images/img.jpg')
    st.image(image, caption='*')
    
    st.subheader('Saved trained models:')
    st.text("ais-dsp-tien/serving/frontend/model")

    
st.title('Diabetes Prediction: Sherlock Holmes')
model = None

model_name = ""
def train_model(data_train: pd.DataFrame, model_class, **model_kwargs):
    X = data_train[['Age','Sex','BMI','BP','S1','S2','S3','S4','S5','S6']]
    y = data_train['Quantitative']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trained_model = model_class(**model_kwargs)
    trained_model.fit(X_train, y_train)
    
    model_name = 'models/' + "diabetes_{}.joblib".format(str(trained_model))
    joblib.dump(trained_model, model_name)
    
    y_pred = trained_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2
    

st.subheader('Training!')
#  CSV file
train_file = st.file_uploader('Choose a CSV file for data train : Full Data')
if train_file:
    st.write('filename: ', train_file.name)
    data_train = pd.read_csv(train_file)
    st.write(data_train)

    st.text("Dimension: " + str(data_train.shape))
    

option_model = st.selectbox('Training model selection: ', ('LinearRegression', 'ElasticNet', 'RandomForestRegressor')) 
alpha = None
l1_ratio = None
if option_model == 'ElasticNet':
    param_cols = st.beta_columns(2)
    with param_cols[0]:
        alpha = st.number_input('alpha', min_value=0.000, max_value=1.000)
    with param_cols[1]:
        l1_ratio = st.number_input('l1_ratio', min_value=0.000, max_value=1.000)  

models = {'LinearRegression': LinearRegression, 'ElasticNet': ElasticNet, 'RandomForestRegressor': RandomForestRegressor}
# training
if st.button('Training'):
    if train_file:
        model_kwargs = {}
        if alpha and l1_ratio:
            model_kwargs = {'alpha': alpha, 'l1_ratio': l1_ratio}
        rmse, mae, r2 = train_model(data_train, models[option_model], **model_kwargs)
        
        st.success('Training Successful! rmse = {}, mae = {}, r2 = {}'.format(rmse, mae, r2))
    else:
        st.warning('You need to upload a CSV file')

selected_model_text = st.subheader('Selected Model: ' + str(model))

if not model:
    model_file = st.file_uploader('Choose your own model:')
    if model_file:
        model = model_file
        selected_model_text.subheader('Selected Model: ' + str(model.name))
 

st.subheader('Diabetes Prediction!')
input_expander = st.beta_expander("One Patient", expanded=True)
with input_expander:
    col1, col2 = st.beta_columns(2)
    with col1:
        age = st.number_input('age')
        sex = st.number_input('sex')
        bmi = st.number_input('bmi')
        bp = st.number_input('bp')
        s1 = st.number_input('s1')
    with col2:
        s2 = st.number_input('s2')
        s3 = st.number_input('s3')
        s4 = st.number_input('s4')
        s5 = st.number_input('s5')
        s6 = st.number_input('s6')

if st.button('Predict for one patient'):

    if not model:
        st.warning("No model existed! Please select your model!")
        st.stop()
    res = requests.post("http://127.0.0.1:8000/predict?age={}&sex={}&bmi={}&bp={}&s1={}&s2={}&s3={}&s4={}&s5={}&s6={}".format(age,sex,bmi,bp,s1,s2,s3,s4,s5,s5))
    
    predictions = res.json()
    st.success(f'Predictions : {predictions}')


#  CSV file

st.subheader('Multi Prediction!')
X_test_file = st.file_uploader('Choose a CSV file for prediction')

col_test, col_pred = st.beta_columns((2, 1))
with col_test:
    if X_test_file:
        st.write('filename: ', X_test_file.name)
        X_test = pd.read_csv(X_test_file)
        st.write(X_test)
with col_pred:
    if X_test_file:
        st.write('Result')
        st_prediction = st.empty()
    
def inference():
    data=X_test.to_json(orient='records', lines=True).split('\n')
    data=[json.loads(i) for i in data if i != '']
    
    url="http://127.0.0.1:8000/predict_obj"
    res = requests.post(url, json=data)            
    
    return res


if st.button('Predict for multi patients'):
    if not model:
        st.warning("No model existed! Please select your model!")
        st.stop()
    if not X_test_file:
        st.warning('Please input the file CSV!')
        st.stop()
        
    res = inference()
    predictions = res.json().split(" ")
    results = [[i, predictions[i]] for i in range(len(predictions))]
    st.success(f'Successful Prediction!')

    st_prediction.write(pd.DataFrame(predictions, columns=['Prediction'])) 

"""
To cover
- please make sure that there are titles (age, sex, bmi, bp, ... ) in the begining of .csv file
- reading csv one time
- executing only if data is loaded
"""