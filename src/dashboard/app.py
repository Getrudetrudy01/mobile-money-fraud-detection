import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Mobile Money Fraud Detection")
st.caption("LightGBM fraud model · cost-based threshold · drift monitoring")

tab_score, tab_models, tab_monitor = st.tabs(
    ["Score a transaction", "Model results", "Monitoring"])


# ---------------------------------------------------------------- 1. Live scoring
with tab_score:
    try:
        health = httpx.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"Scoring API online · model version {health['model_version']}")
        api_ok = True
    except httpx.HTTPError:
        st.error(f"Scoring API not reachable at {API_URL}. Start it with `make serve`.")
        api_ok = False

    if api_ok:
        with st.form("transaction"):
            c1, c2, c3 = st.columns(3)
            txn_type = c1.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])
            amount = c1.number_input("Amount", min_value=0.01, value=500_000.0, step=1_000.0)
            hour = c2.slider("Hour of day", 0, 23, 3)
            old_org = c2.number_input("Sender balance before", min_value=0.0,
                                      value=500_000.0, step=1_000.0)
            old_dest = c3.number_input("Receiver balance before", min_value=0.0,
                                       value=0.0, step=1_000.0)
            with st.expander("Receiver history (optional; would come from a feature store)"):
                h1, h2, h3 = st.columns(3)
                prior_count = h1.number_input("Prior transfers received", min_value=0, value=None)
                prior_sum = h2.number_input("Prior amount received", min_value=0.0, value=None)
                hours_since = h3.number_input("Hours since last received", min_value=0.0, value=None)
            submitted = st.form_submit_button("Score transaction", type="primary")

        if submitted:
            payload = {
                "step": 24 + hour,  # a time on day 1 with the chosen hour
                "type": txn_type,
                "amount": amount,
                "oldbalanceOrg": old_org,
                "oldbalanceDest": old_dest,
                "dest_prior_txn_count": prior_count,
                "dest_prior_amount_sum": prior_sum,
                "dest_hours_since_last": hours_since,
            }
            response = httpx.post(f"{API_URL}/score", json=payload, timeout=10)

            if response.status_code == 422:
                st.error("The API rejected this input:")
                st.json(response.json()["detail"])
            else:
                result = response.json()
                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
                m2.metric("Decision", "🚨 Review / block" if result["is_fraud"] else "✅ Approve")
                m3.metric("Decision threshold", f"{result['threshold']:.2f}")

                st.markdown("**Why?** Top factors. Positive values push towards fraud, "
                            "negative values towards legitimate.")
                factors = pd.DataFrame(result["top_factors"]).set_index("feature")
                st.bar_chart(factors, horizontal=True)


# ---------------------------------------------------------------- 2. Model comparison
with tab_models:
    metrics_path = REPORTS / "metrics.json"
    if not metrics_path.exists():
        st.info("No results yet. Run `make train` first.")
    else:
        metrics = json.loads(metrics_path.read_text())
        table = pd.DataFrame({
            name: {
                "PR-AUC": r["test"]["pr_auc"],
                "ROC-AUC": r["test"]["roc_auc"],
                "Recall @ 0.1% FPR": r["test"]["recall_at_0.1pct_fpr"],
                "Threshold": r["decision"]["threshold"],
                "Recall at threshold": r["decision"]["test_recall"],
                "False alarms": r["decision"]["test_false_alarms"],
                "Fraud value missed": r["decision"]["test_missed_fraud_amount"],
            }
            for name, r in metrics.items()
        }).T

        st.subheader("Test-set results")
        st.dataframe(table.style.format({
            "PR-AUC": "{:.4f}", "ROC-AUC": "{:.4f}", "Recall @ 0.1% FPR": "{:.1%}",
            "Threshold": "{:.3f}", "Recall at threshold": "{:.1%}",
            "False alarms": "{:,.0f}", "Fraud value missed": "{:,.0f}",
        }), use_container_width=True)

        left, right = st.columns(2)
        left.markdown("**PR-AUC** (higher is better)")
        left.bar_chart(table["PR-AUC"])
        right.markdown("**Fraud value missed** (lower is better)")
        right.bar_chart(table["Fraud value missed"])

        st.info("PaySim fraud is near-deterministic: one feature (`takes_full_balance`) "
                "carries ~95% of the model's gain. The **lightgbm_no_drain** ablation is the "
                "more realistic performance estimate.")


# ---------------------------------------------------------------- 3. Monitoring
with tab_monitor:
    summary_path = REPORTS / "monitoring_summary.json"
    report_path = REPORTS / "drift_report.csv"
    if not summary_path.exists():
        st.info("No monitoring report yet. Run `make monitor` first.")
    else:
        summary = json.loads(summary_path.read_text())
        action = summary.get("recommended_action", "OK")
        if action == "OK":
            st.success("Recommended action: OK")
        else:
            st.error(f"Recommended action: **{action}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Alert rate (reference)", f"{summary['reference_alert_rate']:.2%}")
        m2.metric("Alert rate (production)", f"{summary['production_alert_rate']:.2%}",
                  delta=f"{summary['production_alert_rate'] - summary['reference_alert_rate']:+.2%}",
                  delta_color="inverse")
        m3.metric("Low-volume days", len(summary.get("days_with_low_volume", [])))

        report = pd.read_csv(report_path).set_index("day")
        left, right = st.columns(2)
        left.markdown("**Daily volume** (1.0 = normal)")
        left.line_chart(report["volume_ratio"])
        right.markdown("**Daily alert rate**")
        right.line_chart(report["alert_rate"])

        st.image(str(REPORTS / "drift_heatmap.png"), caption="Feature drift (PSI) per day")