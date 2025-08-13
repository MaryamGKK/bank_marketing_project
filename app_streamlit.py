
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bank Marketing — Call Prioritization", layout="wide")

st.title("Bank Marketing — Call Prioritization & ROI Simulator")
st.write("Upload a customer CSV, score with the trained model, and plan your calling strategy.")

# Paths 
PREPROC_PATH = "models/preprocessor.joblib"
MODEL_PATH = "models/best_model_calibrated.joblib"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    pre = joblib.load(PREPROC_PATH)
    mdl = joblib.load(MODEL_PATH)
    return pre, mdl

def safe_transform(pre, df):
    # Align columns: unknown columns are ignored by OneHotEncoder(handle_unknown='ignore').
    # For missing expected columns, we add them with NaN.
    return pre.transform(df)

preprocessor, model = load_artifacts()

st.sidebar.header("Controls")
top_n = st.sidebar.number_input("Top N calls to preview", min_value=10, max_value=10000, value=50, step=10)
target_contracts = st.sidebar.number_input("Target contracts", min_value=1, max_value=10000, value=300, step=10)

uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded is not None:
    df_new = pd.read_csv(uploaded)
    st.write("Data preview", df_new.head())

    X = df_new.copy()
    X_trans = safe_transform(preprocessor, X)
    p_yes = model.predict_proba(X_trans)[:,1]
    out = df_new.copy()
    out["p_yes"] = p_yes
    out = out.sort_values("p_yes", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    out["cum_expected_contracts"] = out["p_yes"].cumsum()
    out["cum_calls"] = out["rank"]

    # KPI Header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average P(yes)", f"{out['p_yes'].mean():.2%}")
    with col2:
        calls_needed = out.loc[out["cum_expected_contracts"] >= target_contracts, "rank"].min()
        st.metric(f"Calls for {target_contracts} (expected)", f"{int(calls_needed) if not np.isnan(calls_needed) else 'Not reached'}")
    with col3:
        st.metric("Top 10 precision (expected)", f"{out.head(10)['p_yes'].mean():.2%}")

    # Table
    st.subheader("Top Leads")
    st.dataframe(out.head(top_n))

    # Charts
    # 1) Contracts vs Calls (expected)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=out["cum_calls"], y=out["cum_expected_contracts"], mode="lines", name="Expected"))
    fig.update_layout(title="Expected Contracts vs Calls", xaxis_title="Calls", yaxis_title="Expected Contracts", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 2) Probability distribution
    fig2 = px.histogram(out, x="p_yes", nbins=50, title="Score Distribution (P(yes))")
    fig2.update_layout(template="plotly_white", xaxis_title="Predicted Probability", yaxis_title="Count")
    st.plotly_chart(fig2, use_container_width=True)

    # Download
    st.download_button("Download Ranked Call List", out.to_csv(index=False).encode("utf-8"), file_name="call_list_scored.csv", mime="text/csv")

else:
    st.info("Please upload a CSV to begin.")
