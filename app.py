import streamlit as st
import tensorflow as tf
import numpy as np


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model(r"./basic_cnn.h5")
    return model

model = load_model()

html_temp = """
    <div style="background-color:black;"><p style="color:white;font-size:40px;padding:9px">Malaria Detection Using Deep Learning</p></div>
    
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.info("Artificial intelligence (AI) and open source tools, technologies, and frameworks are a powerful combination for improving society. 'Health is wealth' is perhaps a cliche, yet it's very accurate! We will use how AI can be leveraged for detecting the deadly disease malaria with a low-cost, effective, and accurate yet open source deep learning solution.")
# st.sidebar.text("Credits to Kabir")
    
st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("atul0402mishra@gmail.com")

file = st.file_uploader('Upload the Cell Image', type=['jpg','png'])
# import cv2
from PIL import Image, ImageOps
def import_and_predict(image_data, model):
    size = (125, 125)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Upload the Cell Image here")
else:
    image  = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image, model)
    if predictions > 0.5:
	    string = "You have Malaria."
	    st.error(string, icon="⚠️")
    else:
	    string = "You Don't have Malaria."
	    st.success(string, icon="✅")
    # class_names = ["have Malaria", "don't have Malaria"]
    # string = "Your report after analyzing your Cell image is you " + class_names[np.argmax(predictions)]
    
    

if st.button("Exit"):
        st.balloons()
