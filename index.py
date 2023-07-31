import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('cats_and_dogs_small_2.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
st.title('Cat and Dog Classification')
st.write('Upload an image and let the model predict if it is a cat or a dog.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    input_data = preprocess_image(uploaded_file)

    # Make predictions
    prediction = model.predict(input_data)
    if prediction[0][0] > 0.5:
        st.write("Prediction: Cat")
    else:
        st.write("Prediction: Dog")
