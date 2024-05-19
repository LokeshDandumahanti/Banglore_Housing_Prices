
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib as plt

# Load the model and columns
lr_clf = joblib.load("C:/Users/vijay/OneDrive/Desktop/Banglore Housing Project/banglore_home_prices_model.pkl")
X_columns = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/Banglore Housing Project/dora.csv")
OHE = pd.read_csv("C:/Users/vijay/OneDrive/Desktop/Banglore Housing Project/B5.csv")
locations = OHE['location'].tolist()

def get_row_by_location(location, csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Find the row where the 'location' column matches the given location
    row = df.loc[df['location'] == location].iloc[0]
    
    # Remove the 'location' column
    row_values = row.drop('location').tolist()
    
    return row_values

st.title('House Price Prediction')

# Sidebar with area selection
sqft = st.slider('Select the area in sq meters:', min_value=0.0, max_value=3000.0, value=100.0)

# Location selection
location = st.selectbox('Select a location:', locations)

# BHK selection
bhk = st.slider('Select BHK (1-5):', min_value=1, max_value=5, value=3)

# Bath selection
bath = st.slider('Select Bathrooms (1-5):', min_value=1, max_value=5, value=2)

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X_columns.columns==location)[0][0]

    x = np.zeros(len(X_columns.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


if st.button('Predict Price'):
    price_prediction = predict_price(location, bhk, bath, sqft)
    price_prediction = -1*price_prediction
    st.success(f'The predicted price is: {price_prediction}')

