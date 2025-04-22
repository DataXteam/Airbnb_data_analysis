# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Load model and preprocessing tools ---
model          = joblib.load(ROOT / "model" / "xgb_price_predictor.joblib")
scaler         = joblib.load(ROOT / "model" / "standard_scaler.joblib")
final_features = joblib.load(ROOT / "model" / "final_feature_list.joblib")

st.title("🏠 Airbnb Price Predictor")
st.markdown("Fill in the details below to estimate the **listing price**")

# --- Collect user inputs for some key features ---
with st.container():
    st.subheader("📋 Listing Details")
    col1, col2 = st.columns(2)

    with col1:
        accommodates = st.slider("Number of Guests", 1, 16, 2)
        bedrooms     = st.slider("Number of Bedrooms", 0, 10, 1)
        bathrooms    = st.slider("Number of Bathrooms", 0.0, 5.0, 1.0, step=0.5)
        property_type = st.selectbox("Property Type",
                                     ["Apartment", "House", "Loft", "Guesthouse", "Other"])
    with col2:
        num_amenities       = st.slider("Number of Amenities", 0, 50, 15)
        review_scores_rating = st.slider("Review Rating", 0.0, 5.0, 4.5, step=0.1)
        room_type            = st.selectbox("Room Type",
                                            ["Entire home/apt", "Private room", "Shared room"])

# --- Encode & prepare input ---
input_dict = {
    'accommodates': accommodates,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'num_amenities': num_amenities,
    'review_scores_rating': review_scores_rating,
    'room_type_Private room': 1 if room_type == "Private room" else 0,
    'room_type_Shared room':  1 if room_type == "Shared room"  else 0,
    'room_type_Entire home/apt': 1 if room_type == "Entire home/apt" else 0
}
for pt in ["Apartment", "House", "Loft", "Guesthouse", "Other"]:
    input_dict[f'property_type_{pt}'] = 1 if property_type == pt else 0

for feat in final_features:
    input_dict.setdefault(feat, 0)          # fill missing with 0

input_df = pd.DataFrame([input_dict])[final_features]
input_scaled = scaler.transform(input_df)

# --- Background and styling ---
st.markdown(
    """
    <style>
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("https://www.hotelduo.cz/userfiles/23-a10ae65a.jpg");
        background-size: cover;
        background-position: center;
        filter: brightness(0.6);
        z-index: -1;
    }

    .stApp {
        background: transparent;
        color: white;
    }

    label {
        font-size: 20px !important;
        font-weight: 600;
        color: white !important;
    }

    .stSlider > div,
    .stSelectbox div[data-baseweb="select"],
    .stNumberInput input,
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        border-radius: 10px;
        padding: 10px;
        font-size: 18px !important;
    }

    .stButton > button {
        background-color: #ff5a5f;
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #e04848;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Make prediction ---
if st.button("Predict Price"):
    raw_pred = model.predict(input_scaled)[0]     # keep the name ‘raw’ for clarity
    price = np.exp(raw_pred)

    st.markdown(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.9);
                    padding: 30px; border-radius: 15px; text-align: center;
                    margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
            <h2 style="color: #ff5a5f;">💰 Estimated Price: ${price:.2f} per night</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----------  DIAGNOSTICS  ----------
    with st.expander("🔍 Prediction Details & Diagnostics"):
        st.subheader("Raw model output")
        st.write(raw_pred)

        st.subheader("Input row (un‑scaled)")
        st.dataframe(input_df.T)

        # Compare feature sets
        expected_cols = getattr(model, "feature_names_in_", final_features)
        missing  = sorted(set(expected_cols) - set(input_df.columns))
        extra    = sorted(set(input_df.columns)  - set(expected_cols))

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Missing columns**")
            st.write(missing if missing else "—")
        with colB:
            st.markdown("**Unexpected columns**")
            st.write(extra if extra else "—")

        # Show scaled values
        st.subheader("Scaled values sent to model")
        scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
        st.dataframe(scaled_df.T)
