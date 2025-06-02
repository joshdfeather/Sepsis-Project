import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import random
import shap
import plotly.graph_objects as go
from matplotlib import colormaps
import matplotlib.pyplot as plt

# --- Load model and data ---
@st.cache_data
def load_model_and_data():
    with open("enriched_data_dict.pkl", "rb") as f:
        data = pickle.load(f)
    model = joblib.load("xgb_model.pkl")
    return model, data

model, enriched_data_dict = load_model_and_data()

# --- Select a random sepsis patient ---
sepsis_keys = [k for k in enriched_data_dict if k.endswith('_sepsis')]
if "selected_key" not in st.session_state:
    st.session_state.selected_key = random.choice(sepsis_keys)
selected_key = st.session_state.selected_key
df = enriched_data_dict[selected_key].copy()
df = df.drop(columns=["SepsisLabel"], errors="ignore")
total_hours = len(df)

# --- Title & Styling ---
st.title("Sepsis Risk Timeline")
st.subheader(f"Patient: `{selected_key}`")
st.subheader(f"Hours: `{total_hours}`")

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: white !important;
        color: black !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: black !important;
    }
    .stDataFrame, .stTable, .css-1d391kg, .css-1offfwp {
        background-color: white !important;
        color: black !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: white !important;
    }
    .modebar {
        background-color: white !important;
    }
    .hoverlayer text {
        fill: black !important;
    }
    button[kind="primary"] {
        background-color: white !important;
        color: black !important;
        border: 1px solid black !important;
        font-weight: bold !important;
    }
    .css-1cpxqw2 {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Predict sepsis probabilities ---
X_all = df.fillna(method='ffill').fillna(0)
probs = model.predict_proba(X_all)[:, 1]

# --- Determine max risk ---
max_prob = np.max(probs)
max_hour = int(np.argmax(probs)) + 1  # +1 for 1-based indexing
first_risk_idx = next((i for i, p in enumerate(probs) if p > 0.25), None)


# --- Color map for Plotly points ---
colorscale = colormaps["RdYlGn_r"]
color_vals = np.clip(probs, 0, 1)
rgb_colors = [
    f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)"
    for r, g, b, _ in [colorscale(val) for val in color_vals]
]

# --- Create Plotly chart with risk threshold shading ---
fig = go.Figure()

# Risk line + markers
fig.add_trace(go.Scatter(
    x=list(range(1, len(probs) + 1)),
    y=probs,
    mode='lines+markers',
    marker=dict(size=5, color=rgb_colors, line=dict(width=1, color='gray')),
    line=dict(color='lightgray'),
    name="Sepsis Risk"
))

# Threshold zones
fig.add_shape(type="rect", x0=0, y0=0.7, x1=total_hours, y1=1,
              fillcolor="rgba(255,0,0,0.08)", line_width=0)
fig.add_shape(type="rect", x0=0, y0=0.25, x1=total_hours, y1=0.7,
              fillcolor="rgba(255,165,0,0.08)", line_width=0)# Vertical line at first risk crossing
# Only show rising risk line if threshold crossed
if first_risk_idx is not None:
    fig.add_vline(
        x=first_risk_idx + 1,
        line_dash="dash",
        line_color="orange",
        annotation_text="↑ Rising Risk",
        annotation_position="top left",
        annotation_font_color="orange"
    )



# Layout and styling
fig.update_layout(
    xaxis=dict(
        title="Hour",
        color='black',
        tickfont=dict(color='black'),
        titlefont=dict(color='black'),
        gridcolor='lightgray',
        zerolinecolor='lightgray',
    ),
    yaxis=dict(
        title="Sepsis Risk",
        range=[0, 1],
        color='black',
        tickfont=dict(color='black'),
        titlefont=dict(color='black'),
        gridcolor='lightgray',
        zerolinecolor='lightgray',
    ),
    font=dict(
        color='black',
        family='Arial',
        size=14
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=400
)

# --- Display chart + metrics side-by-side ---
col1, col2 = st.columns([3, 1], gap="small")
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    # Add vertical spacer to align middle
    st.markdown("#")  # or use multiple if needed: st.markdown("####")

    # Risk Metric
    st.metric("🔺 Max Risk", f"{max_prob*100:.2f}%", help=f"Occurred at hour {max_hour}")

    # Risk Level Label
    if max_prob > 0.7:
        st.markdown("### 🚨 **High Risk**", unsafe_allow_html=True)
    elif max_prob > 0.25:
        st.markdown("### ⚠️ **Moderate Risk**", unsafe_allow_html=True)
    else:
        st.markdown("### ✅ **Low Risk**", unsafe_allow_html=True)
    if st.button("# 🔁"):
        st.session_state.selected_key = random.choice(sepsis_keys)
        st.rerun()


# --- SHAP explanation at max-risk hour ---
explainer = shap.TreeExplainer(model)
max_row = X_all.iloc[[max_hour - 1]]
shap_values = explainer.shap_values(max_row)

shap_row = pd.Series(shap_values[0], index=max_row.columns)
top_features = shap_row.abs().sort_values(ascending=False).head(5).index.tolist()

# --- SHAP bar plot ---
st.markdown(f"#### Top Contributing SHAP Features at Max Hour: {max_hour}")

fig_bar, ax = plt.subplots(figsize=(6, 3.5))
shap_vals_top = shap_row[top_features]
shap_vals_top.plot(kind="barh", ax=ax, color=["green" if v < 0 else "red" for v in shap_vals_top])
ax.invert_yaxis()
ax.set_xlabel("SHAP Value")
ax.set_title("Top 5 Feature Contributions")
st.pyplot(fig_bar)