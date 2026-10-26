import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image
import cv2

st.title("Food Calories & Nutrition Estimator")


def load_my_model():
    return load_model('food_model.h5') 

model = load_my_model()


def predict_food_from_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    class_names = ['Egg', 'Meat', 'Rice', 'Seafood', 'Vegtable-Fruit']

    predicted_index = np.argmax(predictions)
    print('predicted index:', predicted_index)
    
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

def estimate_food_calories(predicted_name):
    df = pd.read_csv('/Users/mishabulhaq/Desktop/Food_Calories_Estimator/Nutrition Sheet - Sheet1 (1).csv') 
    result = df[df['Food Items'].str.upper() == predicted_name.upper()]
    
    if not result.empty:
        return result.iloc[0].to_dict()
    else:
        return "Information not Available"


uploaded_file = st.file_uploader("Upload a picture of the food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = f"temp_{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    predicted_name = predict_food_from_image(image_path)
    st.write(f"**Food found:** {predicted_name}")

    nutrition_info = estimate_food_calories(predicted_name)
    
    st.success(f"{predicted_name} - Nutrition Informations")
    st.write(nutrition_info)