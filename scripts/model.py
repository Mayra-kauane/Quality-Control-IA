from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

def create_random_forest_model(balanced=False):
    rf = RandomForestClassifier(
        random_state=42,
        class_weight='balanced' if balanced else None
    )
    return MultiOutputClassifier(rf)

def create_xgboost_model():
    xgb = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=1,
        random_state=42
    )
    return MultiOutputClassifier(xgb)