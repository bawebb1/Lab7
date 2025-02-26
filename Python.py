{python}
#| label: Python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_excel('AmesHousing.xlsx')

df = df.select_dtypes(include=[np.number]).dropna()  
X = df.drop(columns=['SalePrice'], errors='ignore') 
y = df['SalePrice'] if 'SalePrice' in df.columns else df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train 
model = LinearRegression()
model.fit(X_train, y_train)

# Save 
joblib.dump(model, 'house_price_model.pkl')

# Load 
model = joblib.load('house_price_model.pkl')

# Streamlit 
st.title("Ames Housing Price Predictor")
st.write("Enter the house features below to get a price prediction.")


input_features = {}
for feature in X.columns[:5]:  # Limit to first 5 
    input_features[feature] = st.number_input(f"{feature}", value=float(X_train[feature].median()))

# Convert inputs 
input_df = pd.DataFrame([input_features])

# Predict price
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")

