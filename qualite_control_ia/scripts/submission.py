import pandas as pd
from scripts.preprocessing import padronizar_binario

def gerar_submissao(modelo, df_test, colunas_entrada, label_cols, output_path="final_submission.xlsx"):
    for col in colunas_entrada:
        if df_test[col].dtype == 'object' or df_test[col].apply(lambda x: isinstance(x, str)).any():
            df_test[col] = df_test[col].apply(padronizar_binario)

    X_test = df_test[colunas_entrada]
    Y_test_pred = modelo.predict(X_test)

    df_pred = pd.DataFrame(Y_test_pred, columns=label_cols)
    df_pred['id'] = df_test['id']
    df_submission = df_pred[['id'] + label_cols]
    
    df_submission.to_excel(output_path, index=False)
    return df_submission