from sklearn.model_selection import train_test_split
import pandas as pd

from scripts.preprocessing import load_and_prepare_data
from scripts.model import create_xgboost_model
from scripts.evaluation import evaluate_model, plot_f1_scores
from scripts.submission import generate_submission

def main():
    df_train, label_cols = load_and_prepare_data("data/bootcamp_train.xlsx")
    feature_cols = [col for col in df_train.columns if col not in label_cols + ['id']]
    
    X = df_train[feature_cols]
    Y = df_train[label_cols]
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = create_xgboost_model()
    model.fit(X_train, Y_train)
    
    report_dict = evaluate_model(model, X_val, Y_val, label_cols)
    plot_f1_scores(report_dict, model_name="XGBoost")
    
    df_test = pd.read_excel("data/bootcamp_test.xlsx")
    submission_df = generate_submission(
        model=model,
        df_test=df_test,
        feature_cols=feature_cols,
        label_cols=label_cols,
        output_path="data/final_submission.xlsx"
    )
    print("Submission file saved at data/final_submission.xlsx")

if __name__ == "__main__":
    main()