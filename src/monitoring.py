from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

import pandas as pd



def monitoring():
    reference = pd.read_csv("data/Iris.csv").drop(columns=['Id','Species'])
    current = reference.sample(10,random_state=42)

    report = Report(
        metrics=[DataDriftPreset(drift_share=0.3)]
    )

    report.run(
        reference_data=reference,
        current_data=current
    )

    report.save_html("report/drift-report.html")
    print("report updated")


if __name__ == "__main__":
    monitoring()    