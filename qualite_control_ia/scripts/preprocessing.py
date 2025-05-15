import pandas as pd

def padronizar_binario(valor):
    """Converte textos binários em 0 ou 1."""
    if str(valor).strip().lower() in ['1', 'true', 's', 'sim', 'y', 'yes']:
        return 1
    return 0

def carregar_e_preparar_dados(caminho_arquivo):
    """Carrega e padroniza os dados."""
    df = pd.read_excel(caminho_arquivo)
    label_cols = [col for col in df.columns if col.startswith("falha_")]

    # Padronizar rótulos
    for col in label_cols:
        df[col] = df[col].apply(padronizar_binario)

    # Padronizar colunas de entrada
    entrada_cols = [col for col in df.columns if col not in label_cols + ['id']]
    for col in entrada_cols:
        if df[col].dtype == 'object' or df[col].apply(lambda x: isinstance(x, str)).any():
            df[col] = df[col].apply(padronizar_binario)

    return df, label_cols
