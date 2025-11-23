import streamlit as st
import pickle
from PIL import Image

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide"
)

# --------------------------------
# DARK GLASSMORPHIC + VISIBLE INPUT CSS
# --------------------------------
css_code = """
<style>

/* DARK Futuristic Background */
.stApp {
    background: linear-gradient(145deg, #0b0f19, #111827, #0f172a);
    background-attachment: fixed;
    color: #e5e7eb !important;
}
/* IMAGE FIT INSIDE CARD */
.image-fit img {
    width: 100% !important;    /* Take full card width */
    height: auto !important;   /* Maintain perfect aspect ratio */
    object-fit: cover !important; 
    border-radius: 16px !important;
    display: block;
}


/* HEADER */
.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
    text-shadow: 0 0 18px rgba(56,189,248,0.45);
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}

/* GLASS CARD – DARK MODE */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 22px;
    padding: 26px;
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    backdrop-filter: blur(14px);
}

/* SECTION TITLE */
.section-title {
    font-weight: 700;
    font-size: 22px;
    color: #f8fafc;
    margin-bottom: 0.8rem;
}

/* MAKE INPUT TEXT VISIBLE */
.stNumberInput label {
    color: #e2e8f0 !important;
}

.stNumberInput input {
    background: rgba(0,0,0,0.35) !important;      /* dark transparent */
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: #ffffff !important;                     /* white text */
    border-radius: 10px !important;
    padding: 10px !important;
    font-size: 16px !important;
}

input::placeholder {
    color: #e2e8f0 !important;
    opacity: 0.8;
}

/* BUTTON – Futuristic gradient */
.stButton>button {
    background: linear-gradient(135deg,#38bdf8,#6366f1);
    color: white;
    border-radius: 8px;
    padding: 0.7rem 1.4rem;
    border: none;
    font-weight: 600;
    width: 100%;
    transition: 0.2s ease;
    box-shadow: 0 0 15px rgba(56,189,248,0.5);
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(56,189,248,0.9);
}

/* PREDICTION BOX */
.prediction-box {
    padding: 18px;
    border-radius: 20px;
    margin-top: 20px;
    border: 1px solid rgba(148,163,184,0.35);
    background: rgba(255,255,255,0.08);
    text-align: center;
    font-size: 20px;
    font-weight: 600;
}

/* IMAGE ROUNDING */
img {
    border-radius: 16px;
}

/* Remove Streamlit footer/menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
"""

# Load CSS
st.markdown(css_code, unsafe_allow_html=True)

# --------------------------------
# MAIN APPLICATION
# --------------------------------
def main():

    # HEADER
    st.markdown("<h1 class='main-title'>Lung Cancer Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>AI-powered risk estimation based on lifestyle and environment</p>",
                unsafe_allow_html=True)

    left_col, right_col = st.columns([1.4, 1], vertical_alignment="top")

    # --------------------------------
    # LEFT SIDE – INPUT FORM
    # --------------------------------
    with left_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📝 Enter Patient Details</div>", unsafe_allow_html=True)

        age = st.number_input("Age", min_value=1, max_value=120, value=None)
        cig = st.number_input("Average Cigarettes Per Day", min_value=0, max_value=100)
        area = st.number_input("Quality of Living Area (1–10)", min_value=1, max_value=10)
        alcohol = st.number_input("Average Alcohol Consumption (per day)", min_value=0, max_value=50)

        predict_btn = st.button("🔍 Predict Lung Cancer Risk")

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------
    # RIGHT SIDE – IMAGE
    # --------------------------------
    with right_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        img = Image.open("360_F_327022239_pB6vIiUa5QrEJ4Wl9RyGv3V7ib9Mx7Xc.webp")   # <-- make sure this file exists
        st.markdown("<div class='image-fit'>", unsafe_allow_html=True)
        st.image(img, caption="AI Lung Scan Visualization")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # LOAD MODEL
    model = pickle.load(open("model.sav", "rb"))
    scaler = pickle.load(open("scaler.sav", "rb"))

    # --------------------------------
    # PREDICTION
    # --------------------------------
    if predict_btn:
        features = [age, cig, area, alcohol]
        scaled = scaler.transform([features])
        result = model.predict(scaled)[0]

        if result == 0:
            st.balloons()
            message = "✔ LOW RISK — The person is not likely to suffer lung cancer."
            color = "#16a34a"
        else:
            message = "⚠ HIGH RISK — The person may be at risk of lung cancer."
            color = "#dc2626"

        st.markdown(
            f"<div class='prediction-box' style='color:{color};'>{message}</div>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()


