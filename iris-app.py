import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Function to load the model
@st.cache
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions
def predict(model, features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction  # Return the prediction directly without indexing

# Streamlit UI
def main():
    st.title('Iris Flower Prediction App')
    
    # Load pre-trained model
    model_path = 'iris_model.pkl'  # Change this path to the location of your Pickle file
    model = load_model(model_path)
    
    # Sidebar with user input fields
    st.sidebar.header('User Input Features')
    sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)
    
    # Make prediction
    input_features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict(model, input_features)
    
    # Display prediction
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    st.write('Prediction:', species_mapping[prediction[0]])

if __name__ == '__main__':
    main()
