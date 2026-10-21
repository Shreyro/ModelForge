
"""Data drift monitoring and dashboard generation for ModelForge."""

import html
import os

import pandas as pd
from scipy.stats import ks_2samp

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.pipeline.data_loader import load_and_preprocess_data


def generate_dashboard():
    """Compare training and test data and generate HTML reports."""

    print("Loading reference and comparison datasets...")

    X_train, X_test, y_train, y_test, feature_names = (
        load_and_preprocess_data()
    )

    reference_df = pd.DataFrame(X_train, columns=feature_names)
    current_df = pd.DataFrame(X_test, columns=feature_names)

    # Calculate feature-level drift using the two-sample
    # Kolmogorov-Smirnov test.
    feature_results = []

    # Bonferroni correction reduces false positives when testing
    # many features simultaneously.
    alpha = 0.05 / len(feature_names)

    for feature in feature_names:
        reference_values = reference_df[feature]
        current_values = current_df[feature]

        result = ks_2samp(reference_values, current_values)

        drift_detected = (
            result.pvalue < alpha and result.statistic >= 0.1
        )

        feature_results.append(
            {
                "feature": feature,
                "ks_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "drift_detected": drift_detected,
            }
        )

    results_df = pd.DataFrame(feature_results)
    drifted_count = int(results_df["drift_detected"].sum())
    feature_count = len(feature_names)
    drift_percentage = 100 * drifted_count / feature_count

    if drifted_count:
        overall_status = "DRIFT DETECTED"
        status_color = "#ff7b72"
    else:
        overall_status = "NO SIGNIFICANT DRIFT"
        status_color = "#3fb950"

    # Generate the detailed Evidently report.
    evidently_report = Report(
        metrics=[DataDriftPreset()]
    )
    evidently_report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    os.makedirs("docs", exist_ok=True)

    evidently_path = os.path.join(
        "docs", "evidently_drift_report.html"
    )
    evidently_report.save_html(evidently_path)

    # Build feature rows for the custom dashboard.
    table_rows = []

    for _, row in results_df.sort_values(
        "ks_statistic", ascending=False
    ).iterrows():
        feature_name = html.escape(str(row["feature"]))
        ks_value = row["ks_statistic"]
        p_value = row["p_value"]

        if row["drift_detected"]:
            badge = '<span class="alert">DRIFT</span>'
        else:
            badge = '<span class="healthy">STABLE</span>'

        table_rows.append(
            f"""
            <tr>
                <td>{feature_name}</td>
                <td>{ks_value:.4f}</td>
                <td>{p_value:.3g}</td>
                <td>{badge}</td>
            </tr>
            """
        )

    rows_html = "\n".join(table_rows)

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ModelForge | Data Drift Monitor</title>
<style>
    body {{
        margin: 0;
        padding: 32px;
        background: #0d1117;
        color: #c9d1d9;
        font-family: "Segoe UI", Arial, sans-serif;
    }}
    header {{
        border-bottom: 1px solid #30363d;
        padding-bottom: 20px;
        margin-bottom: 24px;
    }}
    h1 {{ color: #58a6ff; margin-bottom: 8px; }}
    h2 {{ color: #f0f6fc; font-size: 18px; }}
    .muted {{ color: #8b949e; font-size: 13px; }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 24px 0;
    }}
    .card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
    }}
    .value {{
        font-size: 27px;
        font-weight: 700;
        margin: 10px 0;
    }}
    .status {{ color: {status_color}; }}
    .panel {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-top: 24px;
        overflow-x: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
        text-align: left;
        padding: 12px;
        border-bottom: 1px solid #30363d;
        font-size: 13px;
    }}
    th {{ color: #8b949e; }}
    .healthy, .alert {{
        display: inline-block;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: 600;
    }}
    .healthy {{ color: #3fb950; background: #102a1b; }}
    .alert {{ color: #ff7b72; background: #351717; }}
    a {{ color: #58a6ff; }}
    .note {{
        border-left: 3px solid #d29922;
        padding: 12px 16px;
        background: #1f1a10;
        margin-top: 24px;
        line-height: 1.6;
    }}
    @media (max-width: 600px) {{
        body {{ padding: 16px; }}
    }}
</style>
</head>
<body>
<header>
    <h1>⚡ ModelForge Control Center</h1>
    <div class="muted">Data Drift Monitoring · Breast Cancer Wisconsin dataset</div>
</header>

<div class="grid">
    <div class="card">
        <div class="muted">MONITORING STATUS</div>
        <div class="value status">{overall_status}</div>
        <div class="muted">Based on feature-level statistical tests</div>
    </div>
    <div class="card">
        <div class="muted">FEATURES CHECKED</div>
        <div class="value">{feature_count}</div>
        <div class="muted">Numerical input features</div>
    </div>
    <div class="card">
        <div class="muted">FEATURES FLAGGED</div>
        <div class="value">{drifted_count}</div>
        <div class="muted">Using corrected significance threshold</div>
    </div>
    <div class="card">
        <div class="muted">DRIFTED FEATURE SHARE</div>
        <div class="value">{drift_percentage:.1f}%</div>
        <div class="muted">Share of features flagged</div>
    </div>
</div>

<div class="panel">
    <h2>Feature-level drift results</h2>
    <p class="muted">
        Reference: {len(reference_df)} training rows.
        Comparison: {len(current_df)} held-out test rows.
        Bonferroni-adjusted significance level:
        {alpha:.6g}.
    </p>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>KS statistic</th>
                <th>P-value</th>
                <th>Result</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>

<div class="panel">
    <h2>Detailed report</h2>
    <p>Explore Evidently's dataset-level and feature-level drift analysis.</p>
    <a href="evidently_drift_report.html">
        Open Evidently drift report →
    </a>
</div>

<div class="note">
    <strong>Interpretation:</strong> This report compares a training
    split with a held-out test split from the same source dataset.
    It is a reproducible monitoring demonstration, not evidence of
    real production drift. A stable feature distribution also does
    not guarantee stable model accuracy.
</div>

<p class="muted">Generated by ModelForge. No live production telemetry is collected.</p>
</body>
</html>
"""

    dashboard_path = os.path.join("docs", "data_drift_report.html")

    with open(dashboard_path, "w", encoding="utf-8") as file:
        file.write(dashboard_html)

    print("\nDrift monitoring complete.")
    print(f"Features checked: {feature_count}")
    print(f"Features flagged: {drifted_count}")
    print(f"Custom dashboard: {dashboard_path}")
    print(f"Evidently report: {evidently_path}")


if __name__ == "__main__":
    generate_dashboard()
