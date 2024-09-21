import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('output_model.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion MNIST Classifier")
st.markdown("""
    ### Upload an image of a Fashion item to classify
    The model will predict which type of clothing item the image represents.
""")

uploaded_file = st.file_uploader("Choose as image...", type=["png","jpeg","jpg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    st.write(f"Prediction: {class_names[predicted_class]}")

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        text-align: center;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    .stMarkdown {
        font-family: 'Arial';
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)