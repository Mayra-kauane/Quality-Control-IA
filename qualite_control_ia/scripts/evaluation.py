from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(model, X_val, Y_val, label_cols):
    Y_pred = model.predict(X_val)
    report = classification_report(Y_val, Y_pred, target_names=label_cols, output_dict=True)
    print(classification_report(Y_val, Y_pred, target_names=label_cols))
    return report

def plot_f1_scores(report_dict, model_name="Model"):
    f1_scores = {
        k: v["f1-score"]
        for k, v in report_dict.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
    plt.title(f"F1-score per Class - {model_name}")
    plt.ylabel("F1-score")
    plt.xlabel("Failure Class")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
