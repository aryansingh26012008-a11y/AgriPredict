import pickle
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="wide",
)

# -------------------------------------------------
# Custom UI
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #101827;
        color: #f5f7fb;
    }

    .hero {
        display: flex;
        align-items: center;
        gap: 24px;
        padding: 22px;
        border: 1px solid #00c995;
        border-radius: 24px;
        background: #10373a;
        margin-bottom: 35px;
    }

    .hero img {
        width: 120px;
        height: 120px;
        object-fit: cover;
        border-radius: 18px;
    }

    .hero-title {
        color: #00d39f;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-result {
        color: #f1f5f9;
        font-size: 1.55rem;
        font-weight: 750;
    }

    .result-card {
        padding: 24px;
        border: 1px solid #00c995;
        border-radius: 22px;
        background: #10373a;
        margin-top: 25px;
    }

    .result-title {
        color: #00d39f;
        font-size: 1.8rem;
        font-weight: 800;
    }

    .result-subtitle {
        color: #c6d4dc;
        font-size: .9rem;
        margin-top: 6px;
    }

    div[data-testid="stForm"] {
        border: 1px solid #27364d;
        border-radius: 20px;
        padding: 22px;
        background: #121d2d;
    }

    label {
        color: #dbe5ef !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #00b987;
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 12px;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #00d39f;
        color: #07151a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Load the exact model saved by crop.py
# -------------------------------------------------
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(
        "model.pkl was not found. Run your original crop.py first "
        "so it creates model.pkl, then run this app."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load model.pkl: {error}")
    st.stop()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <img src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400"
             alt="Green agricultural field">
        <div>
            <div class="hero-title">Recommended Crop</div>
            <div class="hero-result">Enter field conditions below</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🌾 Soil and Weather Conditions")

# -------------------------------------------------
# Input form
# -------------------------------------------------
with st.form("crop_recommendation_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=78.0,
            step=1.0,
            help="Nitrogen content in the soil",
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=27.87974,
            format="%.5f",
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=280.980,
            format="%.3f",
        )

    with col2:
        P = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=97.0,
            step=1.0,
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=82.0027,
            format="%.4f",
        )

    with col3:
        K = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=80.0,
            step=1.0,
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.563,
            format="%.3f",
        )

    submitted = st.form_submit_button(
        "🌱 Recommend Crop",
        use_container_width=True,
    )

# -------------------------------------------------
# Prediction
# Uses the same feature order as crop.py:
# N, P, K, temperature, humidity, ph, rainfall
# -------------------------------------------------
if submitted:
    features = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ],
    )

    try:
        prediction = model.predict(features)
        result = prediction[0]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">🌱 {str(result).title()}</div>
                <div class="result-subtitle">
                    Recommended crop based on your input values.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# -------------------------------------------------
# Footer note
# -------------------------------------------------
st.markdown("---")
st.caption(
    "This app uses your existing model.pkl created by crop.py. "
    "It does not retrain, rescale, or modify your algorithm."
)
 ##### py -m streamlit run app.py --server.headless true --server.port 8502