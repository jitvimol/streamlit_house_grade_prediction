import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import os
from fastai.vision.all import *

st.sidebar.image("./pics/superai.png", caption=None, width=300, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
st.sidebar.header('👉 SUPERAI2 Observer Home 👈')
st.sidebar.header('Choose Prediction Model')

model_choice = st.sidebar.selectbox('Select Prediction Model', ['House Grade Classification','Properties Tagging','Food prediction'], key='1')

if model_choice == 'House Grade Classification':
    col1,col2,col3 = st.columns(3)
    with col2:
        st.image("./pics/AIAT-NO-BG.png", caption=None, width=200, use_column_width=None, clamp=False, channels='RGB',output_format='auto')
    st.title('🏡 Prediction house grade 🏠')
    # uploaded_files = st.file_uploader("Upload House Pictures", accept_multiple_files=True)

    uploaded_files = st.file_uploader("Upload House Pictures",type=['png','jpeg','jpg'])
    if uploaded_files is not None:
        file_details = {"FileName":uploaded_files.name,"FileType":uploaded_files.type}
        st.write(file_details)
        img = Image.open(uploaded_files)
        st.image(img)
        with open(os.path.join("/home/ubuntu/front-end/uploaded_pics",uploaded_files.name),"wb") as f: 
            f.write(uploaded_files.getbuffer())
            st.success("Pictures saved successfully, ready for prediction!!")
            st.info("Press Predict button below to evaluate the property")

    submit = st.button('👉 Predict 👈')
    if submit:
        # pickle_in = open('/home/ubuntu/front-end/model_1/effnet_bal7200_aug_30ep_lr3.pkl', 'rb')
        # learn = pickle.load(pickle_in)
        st.balloons()
        learn = load_learner('/home/ubuntu/front-end/model_1/effnet_bal7200_aug_30ep_lr3.pkl')
        prediction = learn.predict('/home/ubuntu/front-end/uploaded_pics/'+str(file_details['FileName']))[0]
        if prediction =='0':
            st.error("No house detect in this picture 🤣🤣")
        elif prediction =='1':
            st.success("Above house picture is Grade A+B, RICH PEOPLE!! 💸💰")
            st.text("Expected price is > 30MB")
        elif prediction =='2':
            st.success("🧡Above house picture is Grade C")
            st.text("Expected price is > 10MB")
        elif prediction =='3':
            st.success("🔵Above house picture is Grade D")
            st.text("Expected price is > 5MB")
        elif prediction =='4':
            st.success("📣Above house picture is Grade E")
            st.text("Expected price is > 3MB")
        elif prediction =='5':
            st.success("🐳Above house picture is Grade F")
            st.text("Expected price is < 1MB")

if model_choice == 'Properties Tagging':
    st.warning("Under Development")

if model_choice == 'Food prediction':
    st.warning("Under Development")
    col1,col2,col3 = st.columns(3)
    with col2:
        st.image("./pics/AIAT-NO-BG.png", caption=None, width=200, use_column_width=None, clamp=False, channels='RGB',output_format='auto')
    st.title('🍲🥘Food Prediction🥘🥫')

    uploaded_files = st.file_uploader("Upload Food Pictures",type=['png','jpeg','jpg'])
    if uploaded_files is not None:
        file_details = {"FileName":uploaded_files.name,"FileType":uploaded_files.type}
        st.write(file_details)
        img = Image.open(uploaded_files)
        st.image(img)
        with open(os.path.join("/home/ubuntu/front-end/uploaded_pics",uploaded_files.name),"wb") as f: 
            f.write(uploaded_files.getbuffer())         
            st.success("Pictures saved successfully, ready for prediction!!")
            st.info("Press Predict button below to predict food type")

    submit = st.button('Predict')
    if submit:
        st.warning("Under Development")
        # pickle_in = open('/home/ubuntu/front-end/model_1/effnet_bal7200_aug_30ep_lr3.pkl', 'rb')
        # learn = pickle.load(pickle_in)
        # learn = load_learner('/home/ubuntu/front-end/model_1/effnet_bal7200_aug_30ep_lr3.pkl')
        # prediction = learn.predict('/home/ubuntu/front-end/uploaded_pics/'+str(file_details['FileName']))[0]