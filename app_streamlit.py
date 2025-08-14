import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Bank Marketing Campaign Planner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Setup ---
PREPROC_PATH = Path("./models/preprocessor.pkl")
MODEL_PATH = Path("./models/best_model.pkl")

# --- Helper Functions ---
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with st.spinner("Loading preprocessor and model..."):
        try:
            preprocessor = joblib.load(PREPROC_PATH)
            model = joblib.load(MODEL_PATH)
            st.success("Model artifacts loaded successfully!")
            return preprocessor, model
        except FileNotFoundError as e:
            st.error(f"Error: Required file not found. Please ensure '{e.filename}' exists in the root directory.")
            return None, None

def initial_data_cleaning(df):
    """Performs the same initial cleaning and feature engineering as in notebook 2."""
    
    # Standardize column names
    df.columns = df.columns.str.replace('.', '_').str.replace('-', '_')

    # Drop the 'duration' column if it exists
    df = df.drop('duration', axis=1, errors='ignore')
    
    # Impute 'unknown' values for specific columns with the mode
    for col in ['default', 'housing', 'loan']:
        if col in df.columns:
            # Only impute if 'unknown' is a value, otherwise mode will be wrong
            if 'unknown' in df[col].unique():
                mode_val = df[col].mode()[0]
                df[col] = df[col].replace('unknown', mode_val)

    # Create age_group feature if 'age' column exists
    if 'age' in df.columns:
        bins = [18, 30, 40, 50, 60, 100]
        labels = ['18-29', '30-39', '40-49', '50-59', '60+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    return df

def reindex_and_validate(df, expected_columns):
    """
    Reindexes the DataFrame to match the expected column order and adds missing columns.
    """
    missing_cols = set(expected_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = None  # Add missing columns with None values
    
    # Ensure columns are in the correct order for the preprocessor
    df_reindexed = df[expected_columns]
    
    return df_reindexed

# --- UI Layout ---
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏦 Bank Marketing Campaign Planner</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Prioritize calls to maximize term deposit subscriptions.</h3>", unsafe_allow_html=True)
st.markdown("---")

preprocessor, model = load_artifacts()

if preprocessor and model:
    # Get the feature names the preprocessor was fitted on
    try:
        expected_columns = preprocessor.feature_names_in_
    except AttributeError:
        st.error("Could not retrieve feature names from the preprocessor. Please ensure it was saved correctly.")
        st.stop()
    
    # --- Sidebar Controls ---
    st.sidebar.header("🎯 Campaign Settings")
    st.sidebar.markdown("Adjust the parameters to fit your campaign goals.")
    
    top_n = st.sidebar.slider("Top N Leads to Preview", min_value=10, max_value=500, value=50, step=10)
    target_contracts = st.sidebar.number_input("Target Contracts", min_value=1, max_value=1000, value=300, step=10)
    
    # --- File Uploader and Processing ---
    st.subheader("📁 Upload Your Customer Data")
    st.info("The app is configured to accept the raw, uncleaned data file.")
    uploaded = st.file_uploader("Upload a CSV file containing potential customer data.", type=["csv"])

    if uploaded is not None:
        try:
            # Read the file, assuming ';' as a separator for the original dataset
            df_raw = pd.read_csv(uploaded, sep=';')
            st.write("First 5 rows of your uploaded data:")
            st.dataframe(df_raw.head())

            st.markdown("---")
            st.info("Processing data and generating predictions...")

            with st.spinner("Running predictions..."):
                # Apply initial cleaning and feature engineering
                df_cleaned = initial_data_cleaning(df_raw.copy())

                # Reindex and validate columns before transforming
                df_validated = reindex_and_validate(df_cleaned.copy(), expected_columns)
                
                # Apply preprocessing using the loaded pipeline
                X_processed = preprocessor.transform(df_validated)
                p_yes = model.predict_proba(X_processed)[:, 1]

                out = df_validated.copy()
                out["p_yes"] = p_yes
                out = out.sort_values("p_yes", ascending=False).reset_index(drop=True)
                out["rank"] = out.index + 1
                out["cum_expected_contracts"] = out["p_yes"].cumsum()
                out["cum_calls"] = out["rank"]

                st.success("Predictions complete!")

            # --- Results and KPIs ---
            st.subheader("📊 Key Performance Indicators")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_p_yes = out['p_yes'].mean()
                st.metric("Expected Success Rate", f"{avg_p_yes:.2%}")
            with col2:
                calls_needed_row = out.loc[out["cum_expected_contracts"] >= target_contracts, "rank"].min()
                calls_needed = int(calls_needed_row) if not pd.isna(calls_needed_row) else None
                st.metric(f"Calls for {target_contracts} Contracts (Expected)", f"{calls_needed if calls_needed is not None else 'Not reached'}")
            with col3:
                top_10_precision = out.head(10)['p_yes'].mean()
                st.metric("Top 10 Precision", f"{top_10_precision:.2%}")

            st.markdown("---")
            
            # --- Ranked Leads Table ---
            st.subheader("🏆 Top Leads to Call")
            st.markdown(f"Displaying the top **{top_n}** leads with the highest probability of subscribing.")
            st.dataframe(out.head(top_n))

            with st.expander("Click to view the complete ranked list"):
                st.dataframe(out)

            # --- Charts ---
            st.markdown("---")
            st.subheader("📈 Visualization of Expected Results")
            
            # 1) Contracts vs Calls (expected)
            fig_contracts = go.Figure()
            fig_contracts.add_trace(go.Scatter(x=out["cum_calls"], y=out["cum_expected_contracts"], mode="lines", name="Expected Contracts"))
            fig_contracts.update_layout(
                title="Expected Contracts vs. Number of Calls",
                xaxis_title="Number of Calls",
                yaxis_title="Expected Contracts",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_contracts, use_container_width=True)

            # 2) Probability distribution
            fig_dist = px.histogram(out, x="p_yes", nbins=50, title="Predicted Probability Distribution")
            fig_dist.update_layout(
                template="plotly_white",
                xaxis_title="Predicted Probability (P(yes))",
                yaxis_title="Number of Customers"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # --- Download Button ---
            st.markdown("---")
            st.download_button(
                label="📥 Download Ranked Call List",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="ranked_call_list.csv",
                mime="text/csv"
            )

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.info("Please verify the format and content of your CSV file.")

    else:
        st.info("Please upload a CSV file to get started with the campaign simulation.")