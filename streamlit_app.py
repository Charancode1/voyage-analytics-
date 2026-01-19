import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load models / data
# -----------------------------
flight_model = joblib.load("flight_price_prediction_model.pkl")
gender_model = joblib.load("gender_classification_model.pkl")
hotel_data = joblib.load("hotel_recommender_data.pkl")

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Voyage Analytics", layout="wide")

st.title("✈️ Voyage Analytics Platform")
st.write("Flight Price Prediction • Gender Classification • Hotel Recommendation")

tab1, tab2, tab3 = st.tabs([
    "✈️ Flight Price Prediction",
    "🧑 Gender Classification",
    "🏨 Hotel Recommendation"
])

# =====================================================
# TAB 1: FLIGHT PRICE PREDICTION
# =====================================================
with tab1:
    st.header("✈️ Flight Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        time = st.slider("Flight Time (hours)", 0.5, 5.0, 1.5, step=0.1)
        distance = st.slider("Distance (km)", 100.0, 3000.0, 700.0, step=10.0)

    with col2:
        flight_type = st.selectbox("Flight Type", ["firstClass", "premium"])
        agency = st.selectbox("Agency", ["FlyingDrops", "Rainbow"])

    # One-hot encoding (EXACTLY like training)
    flightType_firstClass = 1 if flight_type == "firstClass" else 0
    flightType_premium = 1 if flight_type == "premium" else 0
    agency_FlyingDrops = 1 if agency == "FlyingDrops" else 0
    agency_Rainbow = 1 if agency == "Rainbow" else 0

    if st.button("Predict Flight Price"):
        X = np.array([[time, distance,
                       flightType_firstClass, flightType_premium,
                       agency_FlyingDrops, agency_Rainbow]])

        price = flight_model.predict(X)[0]

        st.success(f"💰 Estimated Flight Price: ₹ {price:.2f}")

# =====================================================
# TAB 2: GENDER CLASSIFICATION
# =====================================================
with tab2:
    st.header("🧑 Gender Classification")

    col1, col2 = st.columns(2)

    with col1:
        user_code = st.number_input("User Code", min_value=0, value=0)
        age = st.slider("Age", 18, 80, 30)

    with col2:
        company_encoded = st.number_input(
            "Company Encoded Value",
            help="Use the same encoding used during training",
            value=0
        )
        first_name_encoded = st.number_input(
            "First Name Encoded Value",
            help="Encoded first name (LabelEncoder)",
            value=0
        )

    if st.button("Predict Gender"):
        X = np.array([[user_code, age, company_encoded, first_name_encoded]])
        pred = gender_model.predict(X)[0]

        gender_map = {0: "Female", 1: "Male", 2: "Other"}
        st.success(f"🧾 Predicted Gender: **{gender_map.get(pred, 'Unknown')}**")

# =====================================================
# TAB 3: HOTEL RECOMMENDATION
# =====================================================
# ====================================================
# 🏨 HOTEL RECOMMENDATION
# ====================================================
with tabs[2]:
    st.header("🏨 Hotel Recommendation")

    st.caption("Based on historical user booking patterns")

    user_code = st.number_input(
        "User Code",
        min_value=0,
        step=1,
        help="Enter a user code present in the dataset"
    )

    if st.button("Recommend Hotels"):
        if user_code not in hotel_data:
            st.warning("No recommendations found for this user.")
        else:
            recs = hotel_data[user_code]

            # If recommendations are a DataFrame
            if hasattr(recs, "head"):
                st.table(recs.head(5))

            # If recommendations are a list
            elif isinstance(recs, list):
                for i, hotel in enumerate(recs[:5], 1):
                    st.write(f"{i}. {hotel}")

            else:
                st.write(recs)

