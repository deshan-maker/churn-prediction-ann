import json
import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
#MainMenu, footer {visibility: hidden;}

.hero {
    padding: 28px 32px; border-radius: 20px; margin-bottom: 24px;
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 55%, #06B6D4 100%);
    color: white;
}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; color: white;}
.hero p {margin: 6px 0 0; opacity: .9; font-size: 1.02rem;}

.card {
    background: #161B26; border: 1px solid #262D3D; border-radius: 18px;
    padding: 22px 24px; margin-bottom: 18px;
}
.card h3 {margin: 0 0 12px; font-size: 1.05rem; color: #AEB6C8; font-weight: 600;}

.gauge-wrap {text-align: center;}
.gauge-value {font-size: 2.6rem; font-weight: 800; margin-top: -58px;}
.gauge-label {font-size: .95rem; color: #AEB6C8; margin-top: 2px;}

.badge {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-weight: 700; font-size: .9rem; margin-top: 10px;
}
.badge.low {background: rgba(34,197,94,.15); color: #4ADE80;}
.badge.mid {background: rgba(245,158,11,.15); color: #FBBF24;}
.badge.high {background: rgba(239,68,68,.15); color: #F87171;}

.chip {
    display: inline-block; padding: 6px 12px; margin: 4px 6px 4px 0;
    border-radius: 10px; background: #1F2636; border: 1px solid #2B3347;
    font-size: .88rem; color: #D5DAE6;
}
.tip {
    padding: 10px 14px; margin: 6px 0; border-radius: 12px;
    background: #1F2636; border-left: 4px solid #6C63FF; font-size: .92rem;
}
.risk {border-left-color: #F87171;}

[data-testid="stSidebar"] {background: #12161F;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Load model ----------
@st.cache_resource
def load_assets():
    model = load_model("churn_ann.keras")
    scaler = joblib.load("scaler.pkl")
    with open("columns.json") as f:
        columns = json.load(f)
    return model, scaler, columns


model, scaler, columns = load_assets()

YES_NO = ["No", "Yes"]
YES_NO_NOINT = ["No", "Yes", "No internet service"]

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.markdown("## ⚙️ Customer details")

    with st.expander("💳 Account", expanded=True):
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
        total = st.number_input(
            "Total charges ($)", 0.0, 9000.0, float(round(tenure * monthly, 2)), step=10.0
        )
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", YES_NO)
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with st.expander("👤 Customer"):
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", YES_NO)
        partner = st.selectbox("Partner", YES_NO)
        dependents = st.selectbox("Dependents", YES_NO)

    with st.expander("📡 Services"):
        phone = st.selectbox("Phone service", YES_NO)
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online security", YES_NO_NOINT)
        backup = st.selectbox("Online backup", YES_NO_NOINT)
        protection = st.selectbox("Device protection", YES_NO_NOINT)
        support = st.selectbox("Tech support", YES_NO_NOINT)
        tv = st.selectbox("Streaming TV", YES_NO_NOINT)
        movies = st.selectbox("Streaming movies", YES_NO_NOINT)


# ---------- Prediction ----------
def build_row():
    row = {c: 0.0 for c in columns}
    row["SeniorCitizen"] = 1.0 if senior == "Yes" else 0.0
    row["tenure"] = float(tenure)
    row["MonthlyCharges"] = float(monthly)
    row["TotalCharges"] = float(total)

    categorical = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }
    for feature, value in categorical.items():
        name = f"{feature}_{value}"
        if name in row:  # first category of each feature was dropped in training
            row[name] = 1.0
    return pd.DataFrame([row], columns=columns)


prob = float(model.predict(scaler.transform(build_row()), verbose=0).ravel()[0])

if prob < 0.35:
    level, color, text = "low", "#22C55E", "Low risk"
elif prob < 0.5:
    level, color, text = "mid", "#F59E0B", "Medium risk"
else:
    level, color, text = "high", "#EF4444", "High risk"

# ---------- Main layout ----------
st.markdown(
    """
<div class="hero">
<h1>📉 Customer Churn Predictor</h1>
<p>An ANN trained on the Telco dataset estimates how likely a customer is to leave. Adjust the details in the sidebar and watch the risk update live.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="large")

with left:
    arc = 251.3  # length of the semicircle (pi * 80)
    st.markdown(
        f"""
<div class="card gauge-wrap">
<h3>Churn probability</h3>
<svg viewBox="0 0 200 115" width="100%" style="max-width:320px">
<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#262D3D" stroke-width="16" stroke-linecap="round"/>
<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="{prob * arc:.1f} {arc}"/>
</svg>
<div class="gauge-value" style="color:{color}">{prob:.0%}</div>
<div class="badge {level}">{text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    chips = [
        f"🗓 {tenure} months",
        f"💵 ${monthly:.0f}/mo",
        f"📄 {contract}",
        f"🌐 {internet}",
        f"💳 {payment}",
    ]
    st.markdown(
        '<div class="card"><h3>Customer snapshot</h3>'
        + "".join(f'<span class="chip">{c}</span>' for c in chips)
        + "</div>",
        unsafe_allow_html=True,
    )

with right:
    factors = []
    if contract == "Month-to-month":
        factors.append("Month-to-month contract: customers can leave at any time")
    if tenure < 12:
        factors.append("New customer (under 12 months of tenure)")
    if internet == "Fiber optic":
        factors.append("Fiber optic customers churned more often in this dataset")
    if payment == "Electronic check":
        factors.append("Electronic check payers churned more often")
    if security == "No" and internet != "No":
        factors.append("No online security add-on")
    if monthly > 80:
        factors.append("High monthly charges")

    if factors:
        body = "".join(f'<div class="tip risk">⚠️ {f}</div>' for f in factors)
    else:
        body = '<div class="tip">✅ No major risk factors detected for this profile.</div>'
    st.markdown(f'<div class="card"><h3>Possible risk factors</h3>{body}</div>', unsafe_allow_html=True)

    actions = []
    if contract == "Month-to-month":
        actions.append("Offer a discount for switching to a one or two year contract")
    if security == "No" and internet != "No":
        actions.append("Offer a free trial of Online Security or Tech Support")
    if payment == "Electronic check":
        actions.append("Encourage automatic payment (card or bank transfer)")
    if prob >= 0.5:
        actions.append("Have customer service reach out proactively")
    if not actions:
        actions.append("Keep the customer happy with regular check-ins")

    st.markdown(
        '<div class="card"><h3>Suggested retention actions</h3>'
        + "".join(f'<div class="tip">💡 {a}</div>' for a in actions)
        + "</div>",
        unsafe_allow_html=True,
    )

st.caption("Model: ANN (32-16-1, dropout, class weights) · Dataset: Telco Customer Churn · For learning and portfolio use only.")