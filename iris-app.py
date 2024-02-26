import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Function to load and preprocess the Iris dataset
def load_data():
    # Load Iris dataset
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    iris_data = pd.read_csv(url)
    
    # Preprocessing
    X = iris_data.drop('species', axis=1)
    y = iris_data['species']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Function to train and save the model
def train_and_save_model(X_train, y_train):
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open('iris_model.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    return model

# Function to load the model
@st.cache
def load_model():
    with open('iris_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions
def predict(model, features):
    prediction = model.predict(features)
    return prediction

# Streamlit UI
def main():
    st.title('Iris Flower Prediction App')
    
    # Load Iris dataset
    X_train, _, y_train, _ = load_data()
    
    # Train and save the model
    model = train_and_save_model(X_train, y_train)
    
    # Sidebar with user input fields
    st.sidebar.header('User Input Features')
    sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.0)
    
    # Make prediction
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = predict(model, input_features)
    
    # Display prediction
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    if prediction:
        st.write('Prediction:', species_mapping[prediction[0]])

if __name__ == '__main__':
    main()
