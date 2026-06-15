import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from config import (
    DRIFT_REPORT_PATH,
    FEATURES_DELTA_PATH,
    QUARANTINE_DIR,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_PATH
)
from feast_pipeline import get_online_features_dict

METRICS_JSON_PATH = os.path.join(os.path.dirname(MODEL_PATH), "model_metrics.json")

# Streamlit Page Configurations
st.set_page_config(
    page_title="MLOps Feature Store & Drift Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
<style>
    .main { background-color: #0F172A; color: #F1F5F9; }
    .stApp { background-color: #0F172A; }
    .sidebar .sidebar-content { background-color: #1E293B; }
    h1, h2, h3 { color: #38BDF8 !important; font-family: 'Outfit', sans-serif; }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .status-alert {
        background-color: #7F1D1D;
        border-left: 5px solid #EF4444;
        padding: 15px;
        border-radius: 4px;
        color: #FEE2E2;
        margin-bottom: 20px;
    }
    .status-ok {
        background-color: #064E3B;
        border-left: 5px solid #10B981;
        padding: 15px;
        border-radius: 4px;
        color: #D1FAE5;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ MLOps Feature Store & Data Drift Dashboard")
st.caption("PySpark ETL ➔ Delta Lake ➔ Feast Feature Store ➔ Evidently AI ➔ Retraining Loop")

# Sidebar Actions
st.sidebar.image("https://feast.dev/img/logos/feast-horizontal.svg", width=180)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Pipeline Controls")

if st.sidebar.button("🚀 Trigger Feature Pipeline"):
    with st.sidebar.status("Running pipeline..."):
        import subprocess
        # Run orchestrator
        result = subprocess.run(["python3", "src/orchestrator.py"], capture_output=True, text=True)
        st.sidebar.text(result.stdout[-1000:])  # Print tail of output
        if result.returncode == 0:
            st.sidebar.success("Pipeline Run Completed!")
            st.rerun()
        else:
            st.sidebar.error("Pipeline Run Failed!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Pipeline Architecture
- **Processing**: PySpark
- **Offline Storage**: Delta Lake
- **Feature Store**: Feast
- **Online Database**: SQLite
- **Drift Evaluation**: Evidently AI
- **Classifier**: Random Forest
""")

# Load metrics and check drift
drift_file_exists = os.path.exists(DRIFT_REPORT_PATH)
quarantine_exists = os.path.exists(os.path.join(QUARANTINE_DIR, "quarantined_batch.parquet"))

# Layout Column structures
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Pipeline Status")
    
    if quarantine_exists:
        st.markdown(
            '<div class="status-alert"><b>⚠️ ALERT: Drift Detected!</b><br>New production cohort has drifted from the baseline and has been quarantined.</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-ok"><b>✅ SYSTEM STABLE</b><br>No significant drift detected. All feature distributions match reference baseline.</div>', 
            unsafe_allow_html=True
        )
        
    st.subheader("Asset Registry Summary")
    try:
        df = pd.read_parquet(FEATURES_DELTA_PATH)
        baseline_cnt = len(df[df["batch"] == "baseline"])
        prod_cnt = len(df[df["batch"] == "new_production"])
        
        st.metric("Delta Lake Features Row Count", f"{len(df):,}")
        st.markdown(f"- **Baseline (Reference)**: {baseline_cnt:,} snapshots")
        st.markdown(f"- **New Production**: {prod_cnt:,} snapshots")
        
        if quarantine_exists:
            q_df = pd.read_parquet(os.path.join(QUARANTINE_DIR, "quarantined_batch.parquet"))
            st.metric("Quarantined Batch Size", f"{len(q_df):,}", delta="DRIFT TRIGGERED", delta_color="inverse")
    except Exception as e:
        st.error(f"Error reading Delta features: {e}")

# Main Tabs
tab1, tab2, tab3 = st.columns([1, 1, 1]) # We will create dynamic tabs
active_tab = st.radio("Choose Dashboard View", ["📈 Model Comparison", "🔍 Feast Feature Explorer", "🧪 Interactive Drift Report"], horizontal=True)

if active_tab == "📈 Model Comparison":
    st.header("📈 Model Performance under Drift")
    st.markdown("This section shows the performance impact of ignoring feature drift vs retraining the model with the newly arrived cohort.")
    
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r") as f:
            metrics = json.load(f)
            
        m_base = metrics["baseline"]
        m_drift = metrics["drift_ignored"]
        m_retrain = metrics["drift_retrained"]
        
        # Display side-by-side metric changes
        mc1, mc2, mc3 = st.columns(3)
        
        with mc1:
            st.subheader("1. Baseline Performance")
            st.metric("Accuracy", f"{m_base['accuracy']:.3%}")
            st.metric("AUC ROC", f"{m_base['auc_roc']:.4f}")
            st.metric("Precision", f"{m_base['precision']:.3%}")
            st.metric("Recall", f"{m_base['recall']:.3%}")
            
        with mc2:
            st.subheader("2. Drift Ignored")
            acc_diff = m_drift['accuracy'] - m_base['accuracy']
            auc_diff = m_drift['auc_roc'] - m_base['auc_roc']
            st.metric("Accuracy", f"{m_drift['accuracy']:.3%}", delta=f"{acc_diff:.2%}", delta_color="inverse")
            st.metric("AUC ROC", f"{m_drift['auc_roc']:.4f}", delta=f"{auc_diff:.4f}", delta_color="inverse")
            st.metric("Precision", f"{m_drift['precision']:.3%}")
            st.metric("Recall", f"{m_drift['recall']:.3%}")
            
        with mc3:
            st.subheader("3. Drift Retrained")
            acc_recovery = m_retrain['accuracy'] - m_drift['accuracy']
            auc_recovery = m_retrain['auc_roc'] - m_drift['auc_roc']
            st.metric("Accuracy", f"{m_retrain['accuracy']:.3%}", delta=f"{acc_recovery:.2%}")
            st.metric("AUC ROC", f"{m_retrain['auc_roc']:.4f}", delta=f"{auc_recovery:.4f}")
            st.metric("Precision", f"{m_retrain['precision']:.3%}")
            st.metric("Recall", f"{m_retrain['recall']:.3%}")
            
        # Business impact callout
        try:
            df = pd.read_parquet(FEATURES_DELTA_PATH)
            mean_baseline = df[df["batch"] == "baseline"]["annual_income"].mean()
            mean_current = df[df["batch"] == "new_production"]["annual_income"].mean()
            income_mean_drop_pct = abs(mean_baseline - mean_current) / mean_baseline * 100
            st.info(f"💡 **Business Impact summary**: Caught **{income_mean_drop_pct:.1f}% drift** in annual income before model degraded. Retraining recovered **{acc_recovery:.2%} Accuracy** and **{auc_recovery:.4f} AUC ROC**.")
        except:
            pass
            
        # Performance Comparison dataframe
        perf_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC ROC", "Precision", "Recall"],
            "Baseline": [m_base["accuracy"], m_base["auc_roc"], m_base["precision"], m_base["recall"]],
            "Drift Ignored (Prod)": [m_drift["accuracy"], m_drift["auc_roc"], m_drift["precision"], m_drift["recall"]],
            "Drift Caught & Retrained": [m_retrain["accuracy"], m_retrain["auc_roc"], m_retrain["precision"], m_retrain["recall"]]
        })
        st.table(perf_df)
        
    else:
        st.warning("No model performance metrics found. Run the pipeline to populate model training results.")

elif active_tab == "🔍 Feast Feature Explorer":
    st.header("🔍 Low-Latency Feast Client Explorer")
    st.markdown("Retrieve features in real-time from the Feast online store (SQLite) using user entity keys.")
    
    user_input = st.text_input("Enter User Entity ID (e.g. usr_000001 to usr_000600):", "usr_000001")
    
    if st.button("Fetch Real-time Features"):
        try:
            feats = get_online_features_dict([user_input])
            if feats and len(feats) > 0:
                feat_dict = feats[0]
                
                # Format to display nicely
                num_feats_df = pd.DataFrame([
                    {"Feature": key, "Value": val} 
                    for key, val in feat_dict.items() 
                    if key != "user_id"
                ])
                
                st.subheader(f"Online Features for Entity: `{user_input}`")
                st.dataframe(num_feats_df, use_container_width=True)
            else:
                st.warning(f"No features found in Feast online store for user '{user_input}'. Make sure the pipeline has materialized.")
        except Exception as e:
            st.error(f"Error fetching online features: {e}")

elif active_tab == "🧪 Interactive Drift Report":
    st.header("🧪 Evidently AI Interactive Drift Report")
    st.markdown("This interactive report compares baseline dataset distributions against newly arriving production datasets.")
    
    if drift_file_exists:
        with open(DRIFT_REPORT_PATH, 'r', encoding='utf-8') as f:
            html_data = f.read()
            
        # Render the HTML report within a Streamlit iframe
        components.html(html_data, height=900, scrolling=True)
    else:
        st.warning("Evidently drift report HTML file not found. Run the pipeline first to generate it.")
