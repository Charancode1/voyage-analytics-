import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
# Load models / data (PKL only)
# =============================
flight_model = joblib.load("flight_price_prediction_model.pkl")
gender_model = joblib.load("gender_classification_model.pkl")
hotel_data = joblib.load("hotel_recommender_data.pkl")

hotel_stats_df = hotel_data["hotel_stats"]

# =============================
# App config
# =============================
st.set_page_config(
    page_title="Voyage Analytics Platform",
    layout="wide"
)

st.title("✈️ Voyage Analytics Platform")
st.caption(
    "Flight Price Prediction • Gender Classification • Hotel Recommendation"
)

# =============================
# Tabs
# =============================
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
        time = st.slider(
            "Flight Time (hours)",
            min_value=0.5,
            max_value=6.0,
            value=1.5,
            step=0.1
        )

        distance = st.slider(
            "Distance (km)",
            min_value=100.0,
            max_value=3000.0,
            value=700.0,
            step=50.0
        )

    with col2:
        flight_type = st.selectbox(
            "Flight Type",
            ["firstClass", "premium"]
        )

        agency = st.selectbox(
            "Agency",
            ["FlyingDrops", "Rainbow"]
        )

    # One-hot encoding (same as training)
    flightType_firstClass = 1 if flight_type == "firstClass" else 0
    flightType_premium = 1 if flight_type == "premium" else 0
    agency_FlyingDrops = 1 if agency == "FlyingDrops" else 0
    agency_Rainbow = 1 if agency == "Rainbow" else 0

    if st.button("Predict Flight Price"):
        X = np.array([[
            time,
            distance,
            flightType_firstClass,
            flightType_premium,
            agency_FlyingDrops,
            agency_Rainbow
        ]])

        price = flight_model.predict(X)[0]
        st.success(f"💰 Estimated Flight Price: ₹ {price:.2f}")

# =====================================================
# TAB 2: GENDER CLASSIFICATION (FIXED & EXPLAINABLE)
# =====================================================
with tab2:
    st.header("🧑 Gender Classification")
    st.caption("Prediction based on demographic patterns learned from historical users")

    # ---- Dataset-informed limits ----
    MAX_USERS = 1339
    MIN_AGE = 21
    MAX_AGE = 80

    # ---- Company encoding used during training ----
    company_mapping = {
        "4You": 0,
        "Acme Factory": 1,
        "Wonka Company": 2,
        "Monsters CYA": 3,
        "Umbrella LTDA": 4
    }

    col1, col2 = st.columns(2)

    with col1:
        user_code = st.number_input(
            "User Code",
            min_value=0,
            max_value=MAX_USERS,
            value=0,
            help="User ID seen during training (0–1339)"
        )

        age = st.slider(
            "Age",
            min_value=MIN_AGE,
            max_value=MAX_AGE,
            value=30,
            help="Observed age range in training data"
        )

    with col2:
        company = st.selectbox(
            "Company",
            list(company_mapping.keys()),
            help="Company the user belongs to"
        )

        company_encoded = company_mapping[company]

        first_name_encoded = st.slider(
            "First Name Code (encoded)",
            min_value=0,
            max_value=MAX_USERS,
            value=0,
            help=(
                "Numeric encoding of the user's first name "
                "(generated using LabelEncoder during training)"
            )
        )

    if st.button("Predict Gender"):
        X = np.array([[
            user_code,
            age,
            company_encoded,
            first_name_encoded
        ]])

        pred = gender_model.predict(X)[0]

        gender_map = {
            0: "Female",
            1: "Male",
            2: "Other"
        }

        st.success(f"🧾 Predicted Gender: **{gender_map[pred]}**")

# =====================================================
# TAB 3: HOTEL RECOMMENDATION
# =====================================================
with tab3:
    st.header("🏨 Hotel Recommendation")
    st.caption("Personalized suggestions based on booking history")

    user_code = st.number_input(
        "User Code",
        min_value=0,
        max_value=int(hotel_stats_df["userCode"].max()),
        step=1
    )

    top_k = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Recommend Hotels"):
        user_hotels = hotel_stats_df[
            hotel_stats_df["userCode"] == user_code
        ]

        if user_hotels.empty:
            st.warning(
                "No history found. Showing popular hotels instead."
            )

            popular = (
                hotel_stats_df
                .groupby(["place", "name"], as_index=False)
                .agg(
                    bookings=("bookings", "sum"),
                    avg_total_cost=("avg_total_cost", "mean")
                )
                .sort_values("bookings", ascending=False)
                .head(top_k)
            )

            st.dataframe(popular, use_container_width=True)

        else:
            recommendations = (
                user_hotels
                .sort_values(
                    by=["bookings", "avg_total_cost"],
                    ascending=[False, True]
                )
                .head(top_k)
            )

            st.success("Recommended Hotels")
            st.dataframe(
                recommendations[
                    ["place", "name", "bookings", "avg_total_cost"]
                ],
                use_container_width=True
            )
