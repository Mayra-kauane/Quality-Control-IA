import pandas as pd

def standardize_binary(value):
    if str(value).strip().lower() in ['1', 'true', 's', 'sim', 'y', 'yes']:
        return 1
    return 0

def load_and_prepare_data(filepath):
    df = pd.read_excel(filepath)
    label_cols = [col for col in df.columns if col.startswith("falha_")]

    for col in label_cols:
        df[col] = df[col].apply(standardize_binary)

    feature_cols = [col for col in df.columns if col not in label_cols + ['id']]
    for col in feature_cols:
        if df[col].dtype == 'object' or df[col].apply(lambda x: isinstance(x, str)).any():
            df[col] = df[col].apply(standardize_binary)

    return df, label_cols