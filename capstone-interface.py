import streamlit as st 
from streamlit_navigation_bar import st_navbar
import pages as pg
import torch
from transformers import pipeline
import numpy as np
import keras
import json
import pandas as pd
import pages as pg
#pages
pages =["VisusArbiter"]
st.set_page_config(initial_sidebar_state="collapsed", page_title='VisusArbiter', page_icon='⚽')
options = {
    "show_menu": False,
    "show_sidebar": False,
}
styles = {"nav": {"background-color": "rgb(6 56 21)"}}
page = st_navbar(
    pages,
    styles=styles,
    options=options,
)

def page_home():
    st.markdown('''
    <style>
        .stApp{
            background-image: url('https://i.imgur.com/TtS27ya.jpeg');
            background-size: cover;
        }
        .st-emotion-cache-13ln4jf{
            max-width: None;
        }
        .nav[data-v-96be9aef]{
            background-color: rgb(6 56 21);
        }
        button{
            position: fixed; 
            bottom:10%;
        }
        .st-emotion-cache-19rxjzo.ef3psqc12{
            margin-left: 650px;
        }
    </style>
    ''', unsafe_allow_html=True)

    #add button
    page2_btn = st.button('Predict', on_click=page_predict)

def page_predict():
    #markdown
    st.markdown('''
    <style>
        .stApp{
            background-image: url('https://i.imgur.com/TtS27ya.jpeg');
            background-size: cover;
        }
        .st-emotion-cache-13ln4jf{
            max-width: None;
        }
        .nav[data-v-96be9aef]{
            background-color: rgb(6 56 21);
        }
        div[data-testid="column"]:nth-of-type(2) {
            margin-left: 10px;
            margin-top: 10px;
        }
        .stTableStyledTable{
            border: 7px solid black;
            font-weight: bold;
            padding: 5px;
        }
        .stTableStyledTable th {
            border: 7px solid black;
            padding: 5px;
        }
        .stTableStyledTable td {
            border: 7px solid black;
            padding: 5px;
        }
        .row-widget {
            display: flex;
            justify-content: center;
        }
        .st-emotion-cache-19rxjzo:hover{
            border: 1px solid white;
            color: white;
        }
        .ef3psqc12:hover{
            color: white;
        }
        .st-emotion-cache-183lzff.exotz4b0{
            font-family: math;
            font-size: large;
            font-weight: bolder;
        }
        .st-emotion-cache-zuelfj.e1q9reml3{
            height: 107px;
            border-radius: 10px;
        }
        
    </style>
    ''', unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([3, 2])

    def click_button():
        st.session_state.clicked = True

    with col1:
        uploaded_file = st.file_uploader("File upload", type="mp4")
    with col2:
        predict_btn = st.button('Predict', on_click=click_button)

    #dictionaries
    offence_class = {
        0:"No offence", 
        1:"Offence + No card", 
        2:"Offence + Yellow card", 
        3:"Offence + Red card"
    }

    action_class = {
        0:"Tackling", 
        1:"Standing tackling", 
        2:"High leg", 
        3:"Holding", 
        4:"Pushing",
        5:"Elbowing", 
        6:"Challenge", 
        7:"Dive", 
        8:"Dont know"
    }

    #Download model
    pipe_soccer = pipeline("video-classification", model="anirudhmu/videomae-base-finetuned-soccer-action-recognitionx4")

    pipe_soccer.model.classifier = torch.nn.Identity() #removes the final layer
    pipe_soccer.postprocess = lambda x, *a, **k: x.logits #get logits

    offence_model= keras.models.load_model('./multi_foul_model.keras')
    foul_model= keras.models.load_model('./multi_foul_model2.keras')

    #prediction function
    def extract(video_path):
        action_features=[]
        clip_features= pipe_soccer.predict(video_path)
        action_features.append(clip_features)

        features= np.asarray(action_features)
        #prepare data to model
        features= np.reshape(features, (1, 768))

        #predict
        offence_pred= offence_model.predict(features)
        foul_pred= foul_model.predict(features)

        #offence
        offence_max_pred= (offence_pred[0]).max()
        offence_max_index= np.argmax(offence_pred[0])

        #foul
        foul_max_pred= (foul_pred[0]).max()
        foul_max_index= np.argmax(foul_pred[0])

        predictions={
            'offence': [offence_max_pred, offence_class[offence_max_index]],
            'foul': [foul_max_pred, action_class[foul_max_index]]
        }

        return predictions


    #video upload
    path='./SoccerNet/mvfouls/train/action_7/'
    ann = json.load(open('./SoccerNet/mvfouls/train/annotations.json'))
    action_number=7



    #results dictionary
    truth= {
        'action': '',
        'offence': ''
    }
    truth2= {
        'Label': ['Action', 'Offence'],
        'True Value': []
    }
    model_predictions= {
        'action': [],
        'offence': []
    }
    model_predictions2= {
        'Class': [],
        'Confidence': []
    }

    if uploaded_file:
        with col1:
            video_path = path + uploaded_file.name
            st.video(video_path)

    if predict_btn:
        severity= float(ann['Actions'][str(action_number)]['Severity'])
        pred = extract(video_path)

        truth['action']= ann['Actions'][str(action_number)]['Action class']
        truth['offence']= offence_class[int(severity)]
        truth2['True Value'].append(ann['Actions'][str(action_number)]['Action class'])
        truth2['True Value'].append(offence_class[int(severity)])

        model_predictions2['Class'].append(pred['offence'][1])
        model_predictions2['Confidence'].append("{:.2f}".format(pred['offence'][0]))
        model_predictions2['Class'].append(pred['foul'][1])
        model_predictions2['Confidence'].append("{:.2f}".format(pred['foul'][0]))

        with col2:
            st.text('Truth Value')
            st.table(truth2)
            st.text('Predictions')
            st.table(model_predictions2)


if page=='VisusArbiter':
    page_predict()
if page=='Home':
    page_home()