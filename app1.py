import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

@st.cache(allow_output_mutation = True)
def The_Model():
    model = load_model('transfer_model_pneumonia_prediction.h5')
    return model

with st.spinner('Loading the Model into memory...'):
    model = The_Model()

img_size = (150, 150)
last_conv_layer_name = 'mixed7'

st.title('Pneumonia Detection Web App')

def load_image(image_file):
	img = Image.open(image_file).convert("RGB")
	return img

def get_img_array(pil_img, img_size):
    # `img` is a PIL image of size 150x150
    img = pil_img.resize(img_size, Image.ANTIALIAS)
    # `array` is a float32 Numpy array of shape (150, 150, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 150, 150, 3)
    array = np.expand_dims(array, axis=0)
    return array

st.markdown("Welcome to the Pneumonia Detection Web app built with Streamlit")

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

try:
    if image_file is not None:
        # To View Uploaded Image
        st.image(load_image(image_file).resize((250,250), Image.ANTIALIAS), caption = 'Uploaded Image')

    pil_img = load_image(image_file)
    img_array = (get_img_array(pil_img, img_size))/255
    
    
    preds = model.predict(img_array)
    
    pred_class = np.argmax(preds[0])
    if pred_class == 0:
        st.markdown(unsafe_allow_html=True, body="<span style='color:green; font-size: 50px'><strong><h3>Report: Healthy! :smile: </h3></strong></span>")
    else:
        st.markdown(unsafe_allow_html=True, body="<span style='color:red; font-size: 50px'><strong><h4>Report: Pneumonia Affected! :slightly_frowning_face:</h4></strong></span>")
        

    
except:
    pass

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            footer:after {
	            content:'Made by: Abhirup Ghosh | 2022'; 
	            visibility: visible;
                color: black;
	            display: block;
	            position: relative;
	            #background-color: #DAF7A6 ;
	            padding: 5px;
	            top: 2px;
                color: #11FF00;
}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
