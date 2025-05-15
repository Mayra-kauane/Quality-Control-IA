import pandas as pd
from scripts.preprocessing import standardize_binary

def generate_submission(model, df_test, feature_cols, label_cols, output_path="final_submission.xlsx"):
    for col in feature_cols:
        if df_test[col].dtype == 'object' or df_test[col].apply(lambda x: isinstance(x, str)).any():
            df_test[col] = df_test[col].apply(standardize_binary)

    X_test = df_test[feature_cols]
    Y_test_pred = model.predict(X_test)

    df_pred = pd.DataFrame(Y_test_pred, columns=label_cols)
    df_pred['id'] = df_test['id']
    df_submission = df_pred[['id'] + label_cols]

    df_submission.to_excel(output_path, index=False)
    return df_submission