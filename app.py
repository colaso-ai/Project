import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder

# --- Load Model ---
# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# --- Data Preprocessing Steps (to re-create LabelEncoder mappings) ---
# This part mimics the initial data loading and cleaning to get the original categories
# The original notebook loaded from "/content/ML Project.xlsx"
# For Streamlit, we assume this file is available in the same directory as app.py
try:
    original_df = pd.read_excel("ML Project.xlsx") # Assume file is accessible
except FileNotFoundError:
    st.error("ML Project.xlsx not found. Please ensure the data file is in the same directory as app.py.")
    st.stop() # Stop the app if the file is not found

# Rename columns (as in the notebook)
original_df.columns = [
    "Timestamp","Owner","City","Locality","Property_Type",
    "Area_sqft","BHK","Bathrooms","Property_Age",
    "Furnishing","Parking","Price_Lakhs"
]

# Drop non-useful column (as in the notebook)
original_df.drop(columns=["Owner","Timestamp"], inplace=True)

# Area cleaning (as in the notebook)
original_df["Area_sqft"] = pd.to_numeric(original_df["Area_sqft"], errors="coerce")
original_df["Area_sqft"].fillna(original_df["Area_sqft"].median(), inplace=True)

# BHK cleaning (as in the notebook)
original_df["BHK"] = original_df["BHK"].astype(str).str.extract(r'(\d+)').astype(int)

# Bathrooms (as in the notebook)
original_df["Bathrooms"] = pd.to_numeric(original_df["Bathrooms"], errors="coerce")

# Property Age (as in the notebook)
original_df["Property_Age"] = pd.to_numeric(original_df["Property_Age"], errors="coerce")
original_df["Property_Age"].fillna(original_df["Property_Age"].median(), inplace=True)

# Selling Price cleaning (as in the notebook, though not directly used for encoding, it's part of the preproc)
def price_clean(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    nums = re.findall(r'\d+', x)
    if len(nums)==0:
        return np.nan
    val = float(nums[0])
    if "cr" in x or "core" in x:
        return val * 100
    return val

original_df["Price_Lakhs"] = original_df["Price_Lakhs"].apply(price_clean)
original_df["Price_Lakhs"].fillna(original_df["Price_Lakhs"].median(), inplace=True)

# Parking (as in the notebook)
original_df["Parking"] = original_df["Parking"].str.lower().map({"yes":1,"no":0})

# Handle remaining missing values (from the notebook, 'City', 'Locality', 'Property_Type', 'Furnishing', 'Bathrooms')
# Numerical columns -> fill with MEDIAN (Bathrooms was handled above, but re-checking in case)
if original_df["Bathrooms"].isnull().any():
    original_df["Bathrooms"] = original_df["Bathrooms"].fillna(original_df["Bathrooms"].median())

# Categorical columns -> fill with MODE
cat_cols_for_impute = ["City", "Locality", "Property_Type", "Furnishing"]
for col in cat_cols_for_impute:
    if original_df[col].isnull().any():
        original_df[col] = original_df[col].fillna(original_df[col].mode()[0])

# Drop 'Locality' as it was dropped before model training in the notebook
original_df.drop(columns=['Locality'], inplace=True)

# Re-create LabelEncoder instances and fit them using the preprocessed original_df
le_city = LabelEncoder()
le_property_type = LabelEncoder()
le_furnishing = LabelEncoder()

# Fit on unique values to ensure robustness and consistency with training
le_city.fit(original_df["City"].unique())
le_property_type.fit(original_df["Property_Type"].unique())
le_furnishing.fit(original_df["Furnishing"].unique())

# --- Streamlit UI ---
st.set_page_config(page_title="Property Price Prediction", layout="centered")

st.title("🏡 Property Price Prediction App")
st.markdown("Enter the property details to get an estimated selling price.")

with st.sidebar:
    st.header("Input Property Details")

    # Input widgets, using unique values from original_df for select boxes
    city = st.selectbox("City", original_df["City"].unique())
    property_type = st.selectbox("Property Type", original_df["Property_Type"].unique())
    area_sqft = st.number_input("Area (sqft)", min_value=1.0, max_value=10000.0, value=1500.0, format="%.2f")
    bhk = st.selectbox("BHK", sorted(original_df["BHK"].unique()))
    bathrooms = st.selectbox("Bathrooms", sorted(original_df["Bathrooms"].unique()))
    property_age = st.number_input("Property Age (Years)", min_value=0.0, max_value=200.0, value=10.0, format="%.1f")
    furnishing = st.selectbox("Furnishing Status", original_df["Furnishing"].unique())
    parking = st.radio("Parking Available?", ["Yes", "No"])

    predict_button = st.button("Predict Price")

# --- Prediction Logic ---
if predict_button:
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "City": [city],
        "Property_Type": [property_type],
        "Area_sqft": [area_sqft],
        "BHK": [bhk],
        "Bathrooms": [bathrooms],
        "Property_Age": [property_age],
        "Furnishing": [furnishing],
        "Parking": [1 if parking == "Yes" else 0]
    })

    # Apply LabelEncoding to input categorical data
    input_data["City"] = le_city.transform(input_data["City"])
    input_data["Property_Type"] = le_property_type.transform(input_data["Property_Type"])
    input_data["Furnishing"] = le_furnishing.transform(input_data["Furnishing"])

    # Make prediction
    prediction = rf_model.predict(input_data)[0]

    st.subheader("Predicted Property Price")
    st.success(f"The estimated selling price is: ₹ {prediction:,.2f} Lakhs")
    st.balloons()

st.markdown("---")
st.caption("This application uses a Random Forest Regressor model to estimate property prices.")
