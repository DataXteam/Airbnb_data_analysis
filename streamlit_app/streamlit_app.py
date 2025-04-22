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
        bedrooms = st.slider("Number of Bedrooms", 0, 10, 1)
        bathrooms = st.slider("Number of Bathrooms", 0.0, 5.0, 1.0, step=0.5)
        property_type = st.selectbox("Property Type", ["Apartment", "House", "Loft", "Guesthouse", "Other"])

    with col2:
        num_amenities = st.slider("Number of Amenities", 0, 50, 15)
        review_scores_rating = st.slider("Review Rating", 0.0, 5.0, 4.5, step=0.1)
        room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])

# --- Encode & prepare input ---
input_dict = {
    'accommodates': accommodates,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'num_amenities': num_amenities,
    'review_scores_rating': review_scores_rating,
    'room_type_Private room': 1 if room_type == "Private room" else 0,
    'room_type_Shared room': 1 if room_type == "Shared room" else 0,
    'room_type_Entire home/apt': 1 if room_type == "Entire home/apt" else 0
}

# One-hot encode property_type
for pt in ["Apartment", "House", "Loft", "Guesthouse", "Other"]:
    input_dict[f'property_type_{pt}'] = 1 if property_type == pt else 0

means_dict = {
    'host_identity_verified': 0.984296,
    'latitude': 50.081272,
    'longitude': 14.430028,
    'property_type': 7.539669,
    'minimum_nights': 2.538744,
    'maximum_nights': 501.995873,
    'availability_30': 14.605112,
    'availability_365': 186.622077,
    'review_scores_rating': 4.753392,
    'review_scores_location': 4.808961,
    'review_scores_value': 4.713608,
    'instant_bookable': 0.617492,
    'num_amenities': 32.734984,
    'amenity_wifi': 0.988193,
    'amenity_kitchen': 0.930078,
    'amenity_air_conditioning': 0.804906,
    'amenity_heating': 0.929161,
    'amenity_washer': 0.818547,
    'amenity_dryer': 0.899243,
    'amenity_tv': 0.745873,
    'amenity_parking': 0.591586,
    'amenity_pool': 0.022925,
    'amenity_pets_allowed': 0.283815,
    'amenity_long_term_stays_allowed': 0.584709,
    'num_host_verifications': 2.064420,
    'host_duration_days': 2677.098693,
    'has_reviews': 0.916208,
    'calculated_host_listings_count_shared_rooms_log': 0.040534,
    'bathrooms_log': 0.791061,
    'calculated_host_listings_count_private_rooms_log': 0.381652,
    'host_acceptance_rate_log': 0.652217,
    'beds_log': 1.128924,
    'days_since_last_review_log': 4.140207,
    'number_of_reviews_log': 3.293546,
    'bedrooms_log': 0.796955,
    'reviews_per_month_log': 0.925740,
    'accommodates_log': 1.529076,
    'calculated_host_listings_count_entire_homes_log': 1.967396,
    'number_of_reviews_ltm_log': 2.353854,
    'host_response_time_Unknown': 0.061554,
    'host_response_time_days_or_more': 0.014787,
    'host_response_time_within_day': 0.043099,
    'host_response_time_within_hour': 0.821641,
    'host_response_time_within_hours': 0.058918,
    'room_type_Entire_home/apt': 0.845369,
    'room_type_Hotel_room': 0.008597,
    'room_type_Private_room': 0.136291,
    'room_type_Shared_room': 0.009743,
    'neighbourhood_group_Near_Center_East': 0.201169,
    'neighbourhood_group_Near_Center_West_South': 0.089065,
    'neighbourhood_group_New_Town_Vinohrady': 0.179161,
    'neighbourhood_group_North_West_Districts': 0.076685,
    'neighbourhood_group_Old_Town_Center': 0.368294,
    'neighbourhood_group_Outer_Districts': 0.085626
}

# Fill missing features with means_dict (for non-input features)
for feat in final_features:
    if feat not in input_dict:
        input_dict[feat] = means_dict.get(feat, 0)  # Use the mean or 0 if not available in the means dictionary


# Create DataFrame
input_df = pd.DataFrame([input_dict])[final_features]

# Scale the input
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
    log_price_pred = model.predict(input_scaled)[0]
    price = np.exp(log_price_pred)

    # Styled prediction result
    st.markdown(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.9);
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    margin-top: 20px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
            <h2 style="color: #ff5a5f;">💰 Estimated Price: ${price:.2f} per night</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("🔍 Prediction Details"):
        st.write("Log price predicted:", round(log_price_pred, 2))
        st.write("Features used for prediction:")
        st.dataframe(input_df)
