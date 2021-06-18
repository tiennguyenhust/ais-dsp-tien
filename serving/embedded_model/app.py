"""
You need to run the app from the root
To run the app
$ streamlit run serving/embedded_model/app.py
"""

import joblib
import pandas as pd
import streamlit as st
from PIL import Image


st.title('My 1st ML app')
model = joblib.load('../../models/diabetes_model.joblib')


# Upload display image
uploaded_file = st.file_uploader('Choose an image')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)


#  CSV file
csv_file = st.file_uploader('Choose a CSV file')
if csv_file:
    st.write('filename: ', csv_file.name)
    dataframe = pd.read_csv(csv_file)
    st.write(dataframe)
    
    
def inference(data, model_path=None, model=None):
    if model_path:
        model = joblib.load(model_path)
    return model.predict(data)


if st.button('Predict diabetes progression'):
    if csv_file:
        predictions = model.predict(dataframe)
        st.success(f'Predictions : {predictions}')
    else:
        st.warning('You need to upload a CSV file')
        
# One line text
user_input = st.text_input('label goes here', 'one liner')
print(user_input)


# Multi line text
user_input = st.text_area('Your feedback on the model predictions?', 'Empty feedback')
print(user_input)
    

"""
To cover
- reading csv one time
- executing only if data is loaded
"""
    
    
    
    