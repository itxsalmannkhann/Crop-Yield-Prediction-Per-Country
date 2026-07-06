import os
import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------------------------------
# Paths (relative to this script so it runs the same locally and on the cloud)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "Models")
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "yield_df.csv")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    knr = pickle.load(open(os.path.join(MODELS_DIR, "knr.pkl"), "rb"))
    preprocessor = pickle.load(open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "rb"))
    return knr, preprocessor


@st.cache_data
def load_options():
    """Read the dataset once to build dropdown options that match training data."""
    try:
        df = pd.read_csv(DATA_PATH)
        areas = sorted(df["Area"].dropna().unique().tolist())
        items = sorted(df["Item"].dropna().unique().tolist())
        return areas, items
    except Exception:
        return [], []


knr, preprocessor = load_artifacts()
AREAS, ITEMS = load_options()

# Crop emoji lookup for a friendlier result card
CROP_ICONS = {
    "Cassava": "🥔", "Maize": "🌽", "Plantains and others": "🍌",
    "Potatoes": "🥔", "Rice, paddy": "🌾", "Sorghum": "🌾",
    "Soybeans": "🫘", "Sweet potatoes": "🍠", "Wheat": "🌾", "Yams": "🍠",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriYield • Crop Yield Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif;
    }

    /* App background: soft field gradient */
    .stApp {
        background: linear-gradient(160deg, #f3faf1 0%, #eafbe7 40%, #e3f6ff 100%);
    }

    /* Trim default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #1b5e20 0%, #2e7d32 45%, #43a047 100%);
        border-radius: 22px;
        padding: 38px 40px;
        color: #ffffff;
        box-shadow: 0 18px 40px rgba(27, 94, 32, 0.28);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🌾";
        position: absolute;
        font-size: 190px;
        right: -10px;
        top: -30px;
        opacity: 0.12;
        transform: rotate(12deg);
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 10px;
        opacity: 0.92;
        max-width: 640px;
    }
    .hero .pill {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 16px;
        backdrop-filter: blur(6px);
    }

    /* Section card */
    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 15px 15px;
        box-shadow: 0 8px 26px rgba(30, 80, 40, 0.08);
        border: 1px solid rgba(46, 125, 50, 0.10);
        height: 100%;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 0 0 0 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-sub {
        font-size: 0.82rem;
        color: #78909c;
    }

    /* Inputs */
    label, .stNumberInput label, .stSelectbox label {
        font-weight: 500 !important;
        color: #37474f !important;
    }
    div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius: 12px !important;
        border-color: #c8e6c9 !important;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(120deg, #2e7d32, #43a047);
        color: #fff;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 14px;
        padding: 14px 0;
        transition: all 0.2s ease;
        box-shadow: 0 10px 22px rgba(46, 125, 50, 0.30);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(46, 125, 50, 0.42);
        background: linear-gradient(120deg, #1b5e20, #388e3c);
    }

    /* Result card */
    .result {
        background: linear-gradient(120deg, #ffffff, #f1fbef);
        border: 1.5px solid #a5d6a7;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 14px 34px rgba(46, 125, 50, 0.16);
        animation: pop 0.4s ease;
    }
    @keyframes pop {
        0% { transform: scale(0.96); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    .result .big {
        font-size: 3rem;
        font-weight: 800;
        color: #1b5e20;
        line-height: 1.1;
    }
    .result .unit { font-size: 1.1rem; color: #66bb6a; font-weight: 600; }
    .result .label { color: #607d8b; font-size: 0.95rem; margin-top: 6px; }

    /* Metric chips */
    .chip-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }
    .chip {
        background: #f1f8e9;
        border: 1px solid #dcedc8;
        border-radius: 12px;
        padding: 10px 16px;
        font-size: 0.85rem;
        color: #33691e;
        flex: 1;
        min-width: 130px;
        text-align: center;
    }
    .chip b { display: block; font-size: 1.05rem; color: #1b5e20; margin-top: 2px; }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="pill">🌱 Smart Agriculture • Machine Learning</span>
        <h1>AgriYield Predictor</h1>
        <p>Estimate crop yield per country using climate, rainfall and farming
        inputs. Powered by a trained K-Nearest Neighbors model on global
        agricultural data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------------------
# Input layout
# ---------------------------------------------------------------------------
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="card">'
                '<div class="card-title">🌍 Crop & Location</div>'
                '<div class="card-sub">Choose where and what you are growing.</div>'
                , unsafe_allow_html=True)
    
    if AREAS:
        Area = st.selectbox("Country / Area", AREAS, index=AREAS.index("India") if "India" in AREAS else 0)
    else:
        Area = st.text_input("Country / Area", placeholder="e.g. India")

    if ITEMS:
        Item = st.selectbox("Crop", ITEMS, index=ITEMS.index("Maize") if "Maize" in ITEMS else 0)
    else:
        Item = st.text_input("Crop", placeholder="e.g. Maize")

    Year = st.number_input("Year", min_value=1900, max_value=2100, value=2013, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">'\
    '<div class="card-title">🌦️ Environmental Conditions</div>'\
    '<div class="card-sub">Enter the seasonal climate and farming inputs.</div>', 
    unsafe_allow_html=True)


    average_rain_fall_mm_per_year = st.number_input(
        "Average Rainfall (mm / year)", min_value=0.0, value=1485.0, step=1.0
    )
    avg_temp = st.number_input("Average Temperature (°C)", value=16.37, step=0.1)
    pesticides_tonnes = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=121.0, step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Predict action
# ---------------------------------------------------------------------------
btn_col = st.columns([1, 2, 1])[1]
with btn_col:
    predict = st.button("🌾  Predict Crop Yield", use_container_width=True)

if predict:
    if not Area or not Item:
        st.error("Please provide both a country and a crop before predicting.")
    else:
        # Build a named DataFrame in the exact column order the preprocessor
        # was fitted with, so scikit-learn does not raise feature-name warnings.
        features = pd.DataFrame(
            [[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
            columns=[
                "Year",
                "average_rain_fall_mm_per_year",
                "pesticides_tonnes",
                "avg_temp",
                "Area",
                "Item",
            ],
        )
        transformed_features = preprocessor.transform(features)
        prediction = float(knr.predict(transformed_features).reshape(1, -1)[0][0])

        icon = CROP_ICONS.get(Item, "🌾")
        tonnes_per_ha = prediction / 10000.0  # hg/ha -> tonnes/ha

        st.write("")
        st.markdown(
            f"""
            <div class="result">
                <div style="font-size:2.4rem;">{icon}</div>
                <div class="big">{prediction:,.0f} <span class="unit">hg/ha</span></div>
                <div class="label">Predicted yield for <b>{Item}</b> in <b>{Area}</b> ({Year})</div>
                <div class="chip-row">
                    <div class="chip">≈ Tonnes / ha <b>{tonnes_per_ha:,.2f} t</b></div>
                    <div class="chip">Rainfall <b>{average_rain_fall_mm_per_year:,.0f} mm</b></div>
                    <div class="chip">Avg Temp <b>{avg_temp:.1f} °C</b></div>
                    <div class="chip">Pesticides <b>{pesticides_tonnes:,.0f} t</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("hg/ha = hectograms per hectare • 1 tonne = 10,000 hg")
