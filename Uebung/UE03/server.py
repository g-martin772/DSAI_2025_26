import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model-1.keras")
    return model

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


model = load_model()

st.title("This is a cat, trust me!")

uploaded_file = st.file_uploader("Bild auswählen")

button = st.button("Klassifizieren")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild")
    if button:
        st.write("Verarbeite Bild...")
        x = preprocess_image(image)
        pred = model.predict(x)[0][0]
        if pred < 0.5:
            label = "Cat"
        else:
            label = "Dog"
        st.write("Ergebnis:", label)
        st.write("Wahrscheinlichkeit", float(pred))