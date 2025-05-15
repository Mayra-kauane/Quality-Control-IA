from sklearn.model_selection import train_test_split
import pandas as pd

from scripts.preprocessing import carregar_e_preparar_dados
from scripts.model import criar_modelo_xgboost
from scripts.evaluation import avaliar_modelo, plotar_f1_scores
from scripts.submission import gerar_submissao

df_train, label_cols = carregar_e_preparar_dados("data/bootcamp_train.xlsx")
X = df_train.drop(columns=label_cols + ['id'])
Y = df_train[label_cols]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

modelo = criar_modelo_xgboost()
modelo.fit(X_train, Y_train)

report_dict = avaliar_modelo(modelo, X_val, Y_val, label_cols)
plotar_f1_scores(report_dict, modelo_nome="XGBoost")

df_test = pd.read_excel("data/bootcamp_test.xlsx")
df_submission = gerar_submissao(
    modelo=modelo,
    df_test=df_test,
    colunas_entrada=X.columns.tolist(),
    label_cols=label_cols,
    output_path="data/final_submission.xlsx"
)

print("Arquivo de submissão salvo em data/final_submission.xlsx")