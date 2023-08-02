import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('cats_and_dogs_small_2.h5')

# Function to preprocess the image before prediction
def preprocess_image(image):
    image = image.resize((150, 150))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    print(prediction)
    if prediction[0][0] < 0.5:
        return "Cat"
    else:
        return "Dog"

def main():
    st.title("Cat vs Dog Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        prediction = predict(image)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
