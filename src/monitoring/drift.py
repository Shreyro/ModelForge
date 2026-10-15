"""
Data drift monitoring script for ModelForge using Evidently AI.
Compares training reference data against incoming production data to detect drift.
"""

import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.pipeline.data_loader import load_and_preprocess_data

def run_drift_analysis():
    """
    Simulates reference data (training set) vs current production data (with injected drift)
    and generates an Evidently data drift HTML report.
    """
    print("Loading data for drift analysis...")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_preprocess_data()

    # Convert scaled arrays back to DataFrames for Evidently
    reference_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    current_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # Inject deliberate drift into the first feature of the 'current' dataset to simulate real-world changes
    current_df[feature_names[0]] = current_df[feature_names[0]] + 2.5

    print("Running Evidently Data Drift Report...")
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    drift_report.run(reference_data=reference_df, current_data=current_df)

    # Save report output to docs or reports folder
    os_reports_dir = "docs"
    os.makedirs(os_reports_dir, exist_ok=True)
    report_path = os.path.join(os_reports_dir, "data_drift_report.html")
    drift_report.save_html(report_path)

    print(f"Data drift report generated successfully and saved to: {report_path}")

if __name__ == "__main__":
    import os
    run_drift_analysis()