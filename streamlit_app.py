"""
Streamlit UI — Trojan Detection IDS Client.

A frontend that lets the analyst upload a CSV of pre-extracted network
features, choose a classification threshold, and send each row to the
FastAPI backend for analysis. Results are displayed in a colour-coded
table (red = Trojan, green = Benign) alongside a bar chart of model
probabilities for the first analysed row.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Trojan Detection IDS",
    page_icon="🛡️",
    layout="wide",
)

# ── The 15 features expected by the models ────────────────────────
EXPECTED_FEATURES = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s",
]

MODEL_NAMES = [
    "random_forest",
    "lightgbm",
    "xgboost",
    "logistic_regression",
    "mlp",
]

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "mlp": "MLP",
}

API_URL = "http://localhost:8000/analyze"

# ── Custom CSS for a polished look ────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .result-trojan {
        background-color: #fecaca;
        color: #991b1b;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .result-benign {
        background-color: #bbf7d0;
        color: #166534;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛡️ Trojan Detection IDS</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload network-traffic features, set a threshold, '
    'and let 5 ML models classify each packet.</p>',
    unsafe_allow_html=True,
)

# ── Sidebar controls ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        help="A packet is flagged as Trojan if ANY model's probability >= this value.",
    )

    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Upload a CSV with the 15 network features.\n"
        "2. Each row is sent to the FastAPI backend.\n"
        "3. Five models return Trojan probabilities.\n"
        "4. Rows are colour-coded: 🟢 Benign / 🔴 Trojan."
    )

# ── File uploader ─────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a CSV with network-traffic features",
    type=["csv"],
    help=f"The CSV must contain these columns: {', '.join(EXPECTED_FEATURES)}",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns in CSV: **{', '.join(missing)}**")
        st.stop()

    st.success(f"✅ Loaded **{len(df)}** rows with all 15 required features.")

    # Show a preview of the raw data
    with st.expander("📄 Preview uploaded data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # ── Analyse button ────────────────────────────────────────────
    if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0, text="Sending packets to API...")

        for idx, row in df.iterrows():
            features = row[EXPECTED_FEATURES].tolist()
            payload = {"features": features}

            try:
                resp = requests.post(API_URL, json=payload, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                row_result = {"row": idx}
                max_prob = 0.0
                for model_key in MODEL_NAMES:
                    prob = data["results"][model_key]["trojan_probability"]
                    row_result[MODEL_DISPLAY[model_key]] = prob
                    max_prob = max(max_prob, prob)

                row_result["Verdict"] = "🔴 Trojan" if max_prob >= threshold else "🟢 Benign"
                results.append(row_result)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot reach the FastAPI backend at "
                    f"`{API_URL}`. Make sure it is running."
                )
                st.stop()
            except Exception as e:
                st.error(f"❌ Error on row {idx}: {e}")
                st.stop()

            # Update progress bar
            progress.progress(
                (idx + 1) / len(df),
                text=f"Analysing packet {idx + 1} / {len(df)}...",
            )

        progress.empty()

        # ── Build results DataFrame ───────────────────────────────
        results_df = pd.DataFrame(results).set_index("row")

        st.markdown("### 📊 Analysis Results")

        # Colour-code the dataframe
        def highlight_verdict(row):
            """Apply red/green background to each row based on verdict."""
            if "Trojan" in row["Verdict"]:
                return ["background-color: #fecaca; color: #991b1b"] * len(row)
            else:
                return ["background-color: #bbf7d0; color: #166534"] * len(row)

        styled = results_df.style.apply(highlight_verdict, axis=1).format(
            {name: "{:.4f}" for name in MODEL_DISPLAY.values()}
        )
        st.dataframe(styled, use_container_width=True, height=450)

        # ── Summary statistics ────────────────────────────────────
        n_trojan = results_df["Verdict"].str.contains("Trojan").sum()
        n_benign = len(results_df) - n_trojan
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Packets", len(results_df))
        col2.metric("🟢 Benign", n_benign)
        col3.metric("🔴 Trojan", n_trojan)

        # ── Bar chart with row selector ───────────────────────────
        if len(results_df) > 0:
            st.markdown("### 📈 Model Probabilities")
            selected_row = st.number_input(
                "Select packet row to inspect:",
                min_value=0,
                max_value=len(results_df) - 1,
                value=0,
                step=1,
            )
            row_data = results_df.iloc[selected_row]
            model_labels = list(MODEL_DISPLAY.values())
            probs = [row_data[label] for label in model_labels]

            colors = [
                "#ef4444" if p >= threshold else "#22c55e"
                for p in probs
            ]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=model_labels,
                        y=probs,
                        marker_color=colors,
                        text=[f"{p:.4f}" for p in probs],
                        textposition="outside",
                    )
                ]
            )
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"Threshold ({threshold})",
                annotation_position="top left",
            )
            fig.update_layout(
                yaxis_title="Trojan Probability",
                xaxis_title="Model",
                yaxis_range=[0, 1.05],
                template="plotly_white",
                height=420,
                margin=dict(t=40),
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    # Placeholder when no file is uploaded
    st.info("👆 Upload a CSV file to get started.")