# evaluation/save_results.py
# =============================================================
# Step: Results ko CSV mein save karna
# =============================================================
# averaged results dictionary ko ek clean CSV table mein convert
# karta hai, jisse baad mein report/graphs ke liye use kar sakein.

import csv
import os


def save_averaged_results_csv(averaged, output_path):
    """
    averaged results (jaisa full_experiment.py deta hai) ko
    CSV file mein save karta hai.

    CSV columns: Method, Classifier, Metric, Mean, Std
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Classifier", "Metric", "Mean", "Std"])

        for method_name, clf_dict in averaged.items():
            for clf_name, metric_dict in clf_dict.items():
                for metric_name, (mean_val, std_val) in metric_dict.items():
                    writer.writerow([
                        method_name, clf_name, metric_name,
                        round(mean_val, 4), round(std_val, 4)
                    ])

    print(f"Results saved: {output_path}")