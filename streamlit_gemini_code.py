import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="UW WAVE2 Model Monitoring", layout="wide", page_icon="⚙️")

# --- MOCK DATA GENERATION ---
@st.cache_data
def generate_feature_inventory():
    # Base prioritized features
    data = [
        {"Feature Name": "debt_to_income_ratio", "Input Status": "🔴 Detected", "Input Score": 0.182, "SHAP Status": "🔴 Detected", "SHAP Score": 0.140, "Ref Rank": 2, "Group": "Financial", "Type": "Continuous"},
        {"Feature Name": "recent_inquiries_6m", "Input Status": "🔴 Detected", "Input Score": 0.135, "SHAP Status": "🟢 Stable", "SHAP Score": 0.045, "Ref Rank": 5, "Group": "Credit", "Type": "Continuous"},
        {"Feature Name": "co_applicant_credit_score", "Input Status": "🔴 Detected", "Input Score": 0.112, "SHAP Status": "🔴 Detected", "SHAP Score": 0.105, "Ref Rank": 14, "Group": "Credit", "Type": "Continuous"},
        {"Feature Name": "revolving_util_pct", "Input Status": "🟢 Stable", "Input Score": 0.041, "SHAP Status": "🟢 Stable", "SHAP Score": 0.020, "Ref Rank": 8, "Group": "Financial", "Type": "Continuous"},
        {"Feature Name": "monthly_debt_obligations", "Input Status": "🔴 Detected", "Input Score": 0.220, "SHAP Status": "🟢 Stable", "SHAP Score": 0.080, "Ref Rank": 22, "Group": "Financial", "Type": "Continuous"},
        {"Feature Name": "has_medical_debt", "Input Status": "⚪ Failed", "Input Score": None, "SHAP Status": "⚪ Failed", "SHAP Score": None, "Ref Rank": 401, "Group": "Credit", "Type": "Categorical"},
    ]
    
    # Pad to 1300 features to simulate scale
    for i in range(7, 1301):
        data.append({
            "Feature Name": f"feature_{i:04d}",
            "Input Status": "🟢 Stable" if np.random.rand() > 0.03 else "🔴 Detected",
            "Input Score": np.random.uniform(0.01, 0.09),
            "SHAP Status": "🟢 Stable" if np.random.rand() > 0.01 else "🔴 Detected",
            "SHAP Score": np.random.uniform(0.01, 0.09),
            "Ref Rank": i,
            "Group": np.random.choice(["Financial", "Credit", "Demographic", "Behavioral"]),
            "Type": "Continuous"
        })
    return pd.DataFrame(data)

df_features = generate_feature_inventory()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ UW WAVE2 Model Monitoring")
    st.caption("v.2.4.1 | Last Refresh: 2026-11-05 08:00 AM CST")
    st.button("⬇️ Download Report", use_container_width=True)
    
    st.divider()
    st.subheader("📊 Context")
    st.selectbox("Model", ["UW_WAVE2", "UW_WAVE1", "FRAUD_V3"])
    st.selectbox("Version", ["2.4.1", "2.4.0", "2.3.5"])
    
    st.subheader("📅 Time Periods")
    st.selectbox("Reference", ["Baseline (2025 Q4)", "Prior Month (Sep '26)"])
    st.date_input("Monitoring Period", (date(2026, 10, 1), date(2026, 10, 31)))
    
    st.subheader("🔍 Filters")
    st.selectbox("Product", ["All Products", "Auto Loan", "Personal Loan"])
    st.selectbox("Channel", ["All Channels", "Digital", "Branch"])
    st.selectbox("Population", ["All", "New Customers", "Existing Customers"])
    st.button("Apply Filters", type="primary", use_container_width=True)
    
    st.divider()
    st.subheader("📈 Sample Sizes")
    st.text("Reference: 45,000\nCurrent:   14,209")

# --- MAIN HEADER ---
st.title("UW WAVE2 Model Monitoring")
st.caption("Comparison: Baseline (2025 Q4) vs. Current (2026-10-01 to 2026-10-31)")

# Summary Cards
st.error("⚠️ **STATUS: DRIFT DETECTED** | 📱 **APPLICATIONS EVALUATED: 14,209**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Input Drift Flagged", "45 / 1300", "3.46%", delta_color="inverse")
col2.metric("Input Drift Stable", "1,252", "-2", delta_color="normal")
col3.metric("SHAP Drift Flagged", "12 / 1300", "0.92%", delta_color="inverse")
col4.metric("SHAP Drift Stable", "1,285", "0", delta_color="normal")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Feature Inventory", "Feature Explorer", "Run Details"])

with tab1:
    st.subheader("Historical Drift Trends (Past 12 Months)")
    # Generate mock trend data
    months = pd.date_range(start='2025-11-01', periods=12, freq='M').strftime('%b %Y')
    trend_data = pd.DataFrame({
        'Month': months,
        'Input Drift %': np.random.uniform(1.8, 2.5, 11).tolist() + [3.46],
        'SHAP Drift %': np.random.uniform(0.5, 0.9, 11).tolist() + [0.92]
    }).melt('Month', var_name='Metric', value_name='Percentage')
    
    trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
        x=alt.X('Month:N', sort=None),
        y='Percentage:Q',
        color='Metric:N',
        tooltip=['Month', 'Metric', 'Percentage']
    ).properties(height=300)
    st.altair_chart(trend_chart, use_container_width=True)

    st.subheader("Prioritized Findings (Action Required)")
    st.info("Note: Statistical drift indicates a shift in underlying data distributions, not necessarily a reduction in model accuracy. Evaluate business context before retraining.")
    prioritized_df = df_features[(df_features['Input Status'] == '🔴 Detected') | (df_features['SHAP Status'] == '🔴 Detected')].head(5)
    st.dataframe(prioritized_df[['Feature Name', 'Ref Rank', 'Input Status', 'SHAP Status']], hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Feature Inventory (1,300 Features)")
    
    # Inventory Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    search_term = f_col1.text_input("🔍 Search Feature", "")
    status_filter = f_col2.selectbox("Filter Status", ["All", "Flagged for either", "Stable"])
    group_filter = f_col3.selectbox("Group", ["All", "Financial", "Credit", "Demographic", "Behavioral"])
    
    # Filter logic
    filtered_df = df_features.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['Feature Name'].str.contains(search_term, case=False)]
    if status_filter == "Flagged for either":
        filtered_df = filtered_df[(filtered_df['Input Status'] == '🔴 Detected') | (filtered_df['SHAP Status'] == '🔴 Detected')]
    elif status_filter == "Stable":
        filtered_df = filtered_df[(filtered_df['Input Status'] == '🟢 Stable') & (filtered_df['SHAP Status'] == '🟢 Stable')]
    if group_filter != "All":
        filtered_df = filtered_df[filtered_df['Group'] == group_filter]
        
    st.dataframe(filtered_df, hide_index=True, use_container_width=True)

with tab3:
    col_sel, col_btn1, col_btn2 = st.columns([3, 1, 1])
    selected_feature = col_sel.selectbox("Select Feature to Explore", df_features['Feature Name'].tolist())
    
    st.divider()
    
    feature_info = df_features[df_features['Feature Name'] == selected_feature].iloc[0]
    st.markdown(f"### Feature: `{selected_feature}`")
    st.markdown(f"**Rank:** {feature_info['Ref Rank']} | **Type:** {feature_info['Type']} | **Group:** {feature_info['Group']}")
    
    chart_col1, chart_col2 = st.columns(2)
    
    # Helper to generate overlap histograms
    def make_histogram(title, is_drifted, shift=0):
        ref_data = np.random.normal(loc=50, scale=15, size=1000)
        cur_data = np.random.normal(loc=50 + shift, scale=15 if not is_drifted else 18, size=1000)
        
        hist_df = pd.DataFrame({
            'Value': np.concatenate([ref_data, cur_data]),
            'Dataset': ['Reference'] * 1000 + ['Current'] * 1000
        })
        
        chart = alt.Chart(hist_df).mark_area(opacity=0.5, interpolate='step').encode(
            x=alt.X('Value:Q', bin=alt.Bin(maxbins=50)),
            y=alt.Y('count()', stack=None),
            color=alt.Color('Dataset:N', scale=alt.Scale(domain=['Reference', 'Current'], range=['#A9A9A9', '#1f77b4']))
        ).properties(title=title, height=300)
        return chart

    with chart_col1:
        st.markdown("#### PANEL 1: Input Distribution")
        st.markdown(f"**Status:** {feature_info['Input Status']} | **Method:** Wasserstein Distance")
        st.markdown(f"**Score:** {feature_info['Input Score']} (Threshold: 0.100)")
        shift_amount = 10 if feature_info['Input Status'] == '🔴 Detected' else 0
        st.altair_chart(make_histogram("Input Feature Values", feature_info['Input Status'] == '🔴 Detected', shift_amount), use_container_width=True)
        
    with chart_col2:
        st.markdown("#### PANEL 2: SHAP Contribution")
        st.markdown(f"**Status:** {feature_info['SHAP Status']} | **Method:** Wasserstein Distance")
        st.markdown(f"**Score:** {feature_info['SHAP Score']} (Threshold: 0.100)")
        shap_shift = -5 if feature_info['SHAP Status'] == '🔴 Detected' else 0
        st.altair_chart(make_histogram("SHAP Output (Negative = Higher Risk)", feature_info['SHAP Status'] == '🔴 Detected', shap_shift), use_container_width=True)
        
    with st.expander("▶ Expand Technical & Historical Details"):
        st.markdown("""
        **Descriptive Statistics (Current vs Reference):**
        * Mean: Shifted by +12.4%
        * Missing Values: Ref: 0.1% | Cur: 0.2%
        * SHAP Background Data: K-Means summary (k=100) from 2025 Q4
        """)

with tab4:
    st.subheader("Run Metadata")
    st.markdown("""
    * **Run Timestamp:** 2026-11-05 08:00:12 CST
    * **Model Version:** UW_WAVE2 v.2.4.1 (Pickle hash: `a8b9f723901bc`)
    * **SHAP Background Data:** K-Means summary (k=100) from 2025 Q4
    """)
    
    st.subheader("Evaluation Profile")
    st.markdown("""
    * **Numerical Drift Method:** Wasserstein Distance (Threshold: 0.1)
    * **Categorical Drift Method:** Jensen-Shannon Divergence (Threshold: 0.1)
    * **Errors/Failures:** 3 features failed evaluation due to 100% missing rates in the current window (`has_medical_debt`, `legacy_tier`, `promo_code`).
    """)
    
    st.button("⬇️ Download Raw JSON Logs")
    st.button("⬇️ Download Evidently HTML Report")
