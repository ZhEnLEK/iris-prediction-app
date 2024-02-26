# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
@st.cache
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")

# Sidebar - Parameter settings
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

# Main content
st.title('Iris Flower Species Prediction')
st.write('This app predicts the Iris flower species based on input parameters.')

# Load data
df = load_data()

# Sidebar - Data Exploration
st.sidebar.subheader('Data Exploration')
st.sidebar.write('Number of rows:', df.shape[0])
st.sidebar.write('Number of columns:', df.shape[1])
st.sidebar.dataframe(df.head())

# Sidebar - Model Building
st.sidebar.subheader('Model Building')

# Train/test split
train_size = split_size / 100
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model evaluation
st.subheader('Model Test Accuracy Score')
st.write(accuracy_score(y_test, model.predict(X_test)))

# User input for prediction
st.sidebar.subheader('Predict')
input_data = {}
for feature in X.columns:
    value = st.sidebar.slider(f'Input for {feature}', float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_data[feature] = value

# Prediction
prediction = model.predict(pd.DataFrame(input_data, index=[0]))
st.subheader('Prediction')
st.write(prediction[0])
