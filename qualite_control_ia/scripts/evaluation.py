from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def avaliar_modelo(modelo, X_val, Y_val, label_cols):
    Y_pred = modelo.predict(X_val)
    report = classification_report(Y_val, Y_pred, target_names=label_cols, output_dict=True)
    print(classification_report(Y_val, Y_pred, target_names=label_cols))
    return report

def plotar_f1_scores(report_dict, modelo_nome="Modelo"):
    f1_scores = {k: v["f1-score"] for k, v in report_dict.items() if k not in ["accuracy", "macro avg", "weighted avg"]}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
    plt.title(f"F1-score por classe - {modelo_nome}")
    plt.ylabel("F1-score")
    plt.xlabel("Classe de falha")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()