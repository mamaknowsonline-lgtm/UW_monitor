"""
UW Model Monitoring — Streamlit layout
Feature drift + SHAP drift (EvidentlyAI), 1,300 features, per-feature drill-down.

Run:  pip install streamlit evidently pandas numpy plotly scipy
      streamlit run uw_monitoring_app.py

Design notes
- 1,300 features is too many to compute live in the UI. Run drift as a batch job
  (daily/weekly), save results to parquet/DB, and let the app only READ them.
  `compute_drift_with_evidently()` shows the batch logic; the app uses demo data
  until you wire `load_*` functions to your store.
- SHAP drift = run the same Evidently drift test on SHAP-value frames
  (one column per feature) for reference vs current, plus mean|SHAP| shift.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="UW Model Monitoring", layout="wide", page_icon="📊")

N_FEATURES = 1300
PERIODS = pd.date_range("2026-01-01", periods=20, freq="W")


# ----------------------------------------------------------------------------
# Batch job (run offline, not in the app)
# ----------------------------------------------------------------------------
def compute_drift_with_evidently(ref_df, cur_df, ref_shap, cur_shap, features):
    """Returns one row per feature with feature-drift and SHAP-drift stats.
    Evidently >= 0.4 legacy API; for 0.7+ use `from evidently import Report`
    and `from evidently.presets import DataDriftPreset`."""
    from evidently.report import Report
    from evidently.metrics import DataDriftTable

    def _run(ref, cur):
        r = Report(metrics=[DataDriftTable(columns=features)])
        r.run(reference_data=ref[features], current_data=cur[features])
        cols = r.as_dict()["metrics"][0]["result"]["drift_by_columns"]
        return pd.DataFrame(
            [{"feature": k, "score": v["drift_score"], "method": v["stattest_name"],
              "drifted": v["drift_detected"]} for k, v in cols.items()]
        )

    fd = _run(ref_df, cur_df).add_prefix("fd_").rename(columns={"fd_feature": "feature"})
    sd = _run(ref_shap, cur_shap).add_prefix("sd_").rename(columns={"sd_feature": "feature"})
    imp = pd.DataFrame({
        "feature": features,
        "ref_mean_abs_shap": ref_shap[features].abs().mean().values,
        "cur_mean_abs_shap": cur_shap[features].abs().mean().values,
    })
    out = fd.merge(sd, on="feature").merge(imp, on="feature")
    out["ref_rank"] = out["ref_mean_abs_shap"].rank(ascending=False).astype(int)
    out["cur_rank"] = out["cur_mean_abs_shap"].rank(ascending=False).astype(int)
    out["rank_change"] = out["ref_rank"] - out["cur_rank"]
    return out


# ----------------------------------------------------------------------------
# Data loaders — replace demo data with reads from your results store
# ----------------------------------------------------------------------------
@st.cache_data
def load_summary(model, period):
    rng = np.random.default_rng(hash((model, str(period))) % 2**32)
    feats = [f"f_{i:04d}" for i in range(N_FEATURES)]
    groups = rng.choice(["Bureau", "Application", "Banking", "Fraud", "Derived"], N_FEATURES)
    ref_imp = np.sort(rng.pareto(2, N_FEATURES))[::-1] / 10
    cur_imp = ref_imp * rng.lognormal(0, 0.25, N_FEATURES)
    fd = rng.beta(1, 12, N_FEATURES)
    sd = np.clip(fd * rng.uniform(0.3, 1.6, N_FEATURES), 0, 1)
    df = pd.DataFrame({
        "feature": feats, "group": groups,
        "fd_score": fd, "fd_method": "wasserstein", "fd_drifted": fd > 0.1,
        "sd_score": sd, "sd_method": "wasserstein", "sd_drifted": sd > 0.1,
        "ref_mean_abs_shap": ref_imp, "cur_mean_abs_shap": cur_imp,
        "missing_pct": rng.beta(1, 30, N_FEATURES),
    })
    df["ref_rank"] = df["ref_mean_abs_shap"].rank(ascending=False).astype(int)
    df["cur_rank"] = df["cur_mean_abs_shap"].rank(ascending=False).astype(int)
    df["rank_change"] = df["ref_rank"] - df["cur_rank"]
    return df


@st.cache_data
def load_trend(model):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "period": PERIODS,
        "share_feature_drift": np.clip(0.04 + np.cumsum(rng.normal(0.003, 0.01, len(PERIODS))), 0, 1),
        "share_shap_drift": np.clip(0.02 + np.cumsum(rng.normal(0.002, 0.008, len(PERIODS))), 0, 1),
        "volume": rng.integers(8000, 12000, len(PERIODS)),
    })


@st.cache_data
def load_feature_detail(model, feature):
    rng = np.random.default_rng(int(feature[2:]))
    shift = rng.normal(0, 0.4)
    ref_x, cur_x = rng.normal(0, 1, 5000), rng.normal(shift, 1.1, 5000)
    ref_s = 0.3 * ref_x + rng.normal(0, 0.05, 5000)
    cur_s = 0.3 * cur_x * (1 + shift / 3) + rng.normal(0, 0.05, 5000)
    trend = pd.DataFrame({
        "period": PERIODS,
        "feature_drift": np.abs(np.cumsum(rng.normal(0, 0.02, len(PERIODS)))),
        "shap_drift": np.abs(np.cumsum(rng.normal(0, 0.015, len(PERIODS)))),
        "mean_abs_shap": np.abs(0.2 + np.cumsum(rng.normal(0, 0.01, len(PERIODS)))),
    })
    return ref_x, cur_x, ref_s, cur_s, trend


# ----------------------------------------------------------------------------
# Sidebar — global controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 UW Monitoring")
    model = st.selectbox("Model", ["UW_PD_v3.2 (prod)", "UW_PD_v3.3 (challenger)"])
    period = st.selectbox("Current window", PERIODS[::-1], format_func=lambda d: f"Week of {d:%Y-%m-%d}")
    st.caption("Reference: Dev sample (2025-Q1 → Q3)")
    segment = st.multiselect("Segment", ["Prime", "Near-prime", "Subprime", "New-to-credit"])
    st.divider()
    st.subheader("Thresholds")
    fd_thr = st.slider("Feature drift threshold", 0.0, 0.5, 0.10, 0.01)
    sd_thr = st.slider("SHAP drift threshold", 0.0, 0.5, 0.10, 0.01)
    top_k = st.number_input("Top-K important features", 10, N_FEATURES, 100, 10)
    st.divider()
    st.caption("Last batch run: 2026-11-04 06:00 UTC · Evidently 0.4.x")

df = load_summary(model, period)
df["fd_drifted"] = df["fd_score"] > fd_thr
df["sd_drifted"] = df["sd_score"] > sd_thr
trend = load_trend(model)
top = df.nsmallest(top_k, "ref_rank")

# ----------------------------------------------------------------------------
# Header KPIs
# ----------------------------------------------------------------------------
st.header(f"{model} — week of {period:%Y-%m-%d}")
k = st.columns(6)
k[0].metric("Features", f"{N_FEATURES:,}")
k[1].metric("Feature drift", f"{df.fd_drifted.sum()}", f"{df.fd_drifted.mean():.1%} of features", delta_color="inverse")
k[2].metric("SHAP drift", f"{df.sd_drifted.sum()}", f"{df.sd_drifted.mean():.1%} of features", delta_color="inverse")
k[3].metric(f"Drifted in top-{top_k}", f"{(top.fd_drifted | top.sd_drifted).sum()}")
k[4].metric("Top-10 rank changes", f"{(top.nsmallest(10, 'ref_rank').rank_change != 0).sum()}")
status = "🔴 Alert" if df.sd_drifted.mean() > 0.1 else "🟡 Watch" if df.fd_drifted.mean() > 0.05 else "🟢 Stable"
k[5].metric("Status", status)

tab_over, tab_fd, tab_sd, tab_drill = st.tabs(
    ["Overview", "Feature drift", "SHAP drift", "🔍 Feature drill-down"]
)

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tab_over:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(trend, x="period", y=["share_feature_drift", "share_shap_drift"],
                      title="Share of drifted features over time", markers=True)
        fig.update_layout(yaxis_tickformat=".0%", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        by_g = df.groupby("group")[["fd_drifted", "sd_drifted"]].mean().reset_index()
        fig = px.bar(by_g, x="group", y=["fd_drifted", "sd_drifted"], barmode="group",
                     title="Drift rate by feature group")
        fig.update_layout(yaxis_tickformat=".0%", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature drift vs SHAP drift")
    st.caption("Top-right = input shifted AND model reliance shifted → investigate first. "
               "Bubble size = current mean |SHAP|.")
    fig = px.scatter(df, x="fd_score", y="sd_score", size="cur_mean_abs_shap", color="group",
                     hover_name="feature", opacity=0.6, height=450)
    fig.add_vline(fd_thr, line_dash="dash"); fig.add_hline(sd_thr, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Priority list (drifted × important)")
    pri = df[(df.fd_drifted | df.sd_drifted)].assign(
        priority=lambda d: (d.fd_score + d.sd_score) * d.cur_mean_abs_shap
    ).nlargest(20, "priority")
    st.dataframe(pri[["feature", "group", "fd_score", "sd_score", "ref_rank", "cur_rank", "priority"]],
                 use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Shared: filterable table for 1,300 features
# ----------------------------------------------------------------------------
def feature_table(data, score_col, key):
    f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
    q = f1.text_input("Search feature", key=f"q_{key}")
    groups = f2.multiselect("Group", sorted(data.group.unique()), key=f"g_{key}")
    only_drift = f3.toggle("Drifted only", True, key=f"d_{key}")
    only_top = f4.toggle(f"Top-{top_k} only", False, key=f"t_{key}")
    d = data
    if q: d = d[d.feature.str.contains(q, case=False)]
    if groups: d = d[d.group.isin(groups)]
    if only_drift: d = d[d[score_col.replace("score", "drifted")]]
    if only_top: d = d[d.ref_rank <= top_k]
    st.caption(f"{len(d):,} features")
    sel = st.dataframe(
        d.sort_values(score_col, ascending=False), use_container_width=True, hide_index=True,
        height=420, on_select="rerun", selection_mode="single-row", key=f"tbl_{key}",
        column_config={
            score_col: st.column_config.ProgressColumn("Drift score", min_value=0, max_value=1, format="%.3f"),
            "missing_pct": st.column_config.NumberColumn("Missing %", format="%.1%%"),
        },
    )
    rows = sel.selection.rows
    if rows:
        st.session_state["drill_feature"] = d.sort_values(score_col, ascending=False).iloc[rows[0]].feature
        st.info(f"Selected **{st.session_state['drill_feature']}** → open the Feature drill-down tab")


# ----------------------------------------------------------------------------
# Feature drift tab
# ----------------------------------------------------------------------------
with tab_fd:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="fd_score", nbins=60, title="Distribution of drift scores (all features)")
                    .add_vline(fd_thr, line_dash="dash"), use_container_width=True)
    c2.plotly_chart(px.bar(df.nlargest(25, "fd_score"), x="fd_score", y="feature", orientation="h",
                           color="group", title="Top 25 drifted features")
                    .update_yaxes(autorange="reversed"), use_container_width=True)
    feature_table(df[["feature", "group", "fd_score", "fd_method", "fd_drifted", "ref_rank", "missing_pct"]],
                  "fd_score", "fd")


# ----------------------------------------------------------------------------
# SHAP drift tab
# ----------------------------------------------------------------------------
with tab_sd:
    c1, c2 = st.columns(2)
    t20 = df.nsmallest(20, "ref_rank").melt(id_vars="feature", value_vars=["ref_mean_abs_shap", "cur_mean_abs_shap"])
    c1.plotly_chart(px.bar(t20, x="value", y="feature", color="variable", barmode="group", orientation="h",
                           title="Mean |SHAP|: reference vs current (top 20)")
                    .update_yaxes(autorange="reversed"), use_container_width=True)
    movers = df[df.ref_rank <= top_k].reindex(df.rank_change.abs().sort_values(ascending=False).index).dropna().head(20)
    c2.plotly_chart(px.bar(movers, x="rank_change", y="feature", orientation="h",
                           title=f"Biggest importance-rank movers (top-{top_k})")
                    .update_yaxes(autorange="reversed"), use_container_width=True)
    feature_table(df[["feature", "group", "sd_score", "sd_method", "sd_drifted", "ref_mean_abs_shap",
                      "cur_mean_abs_shap", "ref_rank", "cur_rank", "rank_change"]], "sd_score", "sd")


# ----------------------------------------------------------------------------
# Drill-down tab
# ----------------------------------------------------------------------------
with tab_drill:
    default = st.session_state.get("drill_feature", df.nlargest(1, "fd_score").feature.iloc[0])
    feat = st.selectbox("Feature", df.feature, index=int(df.index[df.feature == default][0]))
    row = df[df.feature == feat].iloc[0]
    ref_x, cur_x, ref_s, cur_s, ftrend = load_feature_detail(model, feat)

    m = st.columns(5)
    m[0].metric("Feature drift", f"{row.fd_score:.3f}", "drifted" if row.fd_drifted else "ok", delta_color="off")
    m[1].metric("SHAP drift", f"{row.sd_score:.3f}", "drifted" if row.sd_drifted else "ok", delta_color="off")
    m[2].metric("Mean |SHAP|", f"{row.cur_mean_abs_shap:.4f}", f"{row.cur_mean_abs_shap - row.ref_mean_abs_shap:+.4f}")
    m[3].metric("Importance rank", int(row.cur_rank), int(row.rank_change))
    m[4].metric("Missing %", f"{row.missing_pct:.1%}")

    st.markdown("#### Feature drift")
    c1, c2 = st.columns(2)
    h = go.Figure([go.Histogram(x=ref_x, name="Reference", opacity=0.6, histnorm="probability"),
                   go.Histogram(x=cur_x, name="Current", opacity=0.6, histnorm="probability")])
    c1.plotly_chart(h.update_layout(barmode="overlay", title="Distribution: reference vs current"),
                    use_container_width=True)
    c2.plotly_chart(px.line(ftrend, x="period", y="feature_drift", markers=True, title="Feature drift score over time")
                    .add_hline(fd_thr, line_dash="dash"), use_container_width=True)

    st.markdown("#### SHAP drift")
    c1, c2, c3 = st.columns(3)
    h = go.Figure([go.Histogram(x=ref_s, name="Reference", opacity=0.6, histnorm="probability"),
                   go.Histogram(x=cur_s, name="Current", opacity=0.6, histnorm="probability")])
    c1.plotly_chart(h.update_layout(barmode="overlay", title="SHAP value distribution"), use_container_width=True)
    idx = np.random.default_rng(0).choice(5000, 1500, replace=False)
    dep = pd.DataFrame({"x": np.r_[ref_x[idx], cur_x[idx]], "shap": np.r_[ref_s[idx], cur_s[idx]],
                        "sample": ["Reference"] * 1500 + ["Current"] * 1500})
    c2.plotly_chart(px.scatter(dep, x="x", y="shap", color="sample", opacity=0.4,
                               title="Dependence: feature value vs SHAP"), use_container_width=True)
    c3.plotly_chart(px.line(ftrend, x="period", y=["shap_drift", "mean_abs_shap"], markers=True,
                            title="SHAP drift & importance over time"), use_container_width=True)

    with st.expander("Full Evidently report for this feature"):
        st.caption("Generate in batch: Report(metrics=[ColumnDriftMetric(column_name=feat)]).save_html(path), "
                   "then render with st.components.v1.html(open(path).read(), height=800, scrolling=True)")
