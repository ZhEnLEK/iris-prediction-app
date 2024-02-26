import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load pre-trained model
@st.cache
def load_model():
    return RandomForestClassifier()

model = load_model()

# Define prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    return prediction

# Streamlit UI
st.title('Iris Flower Prediction App')
st.sidebar.header('User Input Features')

# User input fields
sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

# Make prediction
prediction = predict(sepal_length, sepal_width, petal_length, petal_width)

# Display prediction
st.write('Prediction:', prediction)
