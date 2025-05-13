import pandas as pd
import xgboost as xgb
import optuna
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# ✅ Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Load dataset
train_df = pd.read_excel('Training_Dataset.xlsx')

X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder
joblib.dump(label_encoder, 'model/label_encoder.pkl')

# Save feature names
joblib.dump(list(X.columns), 'model/features.pkl')

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'multi:softprob',
        'num_class': len(set(y_encoded)),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, preds)
    return 1.0 - accuracy

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best Parameters:", study.best_params)

# Train final model on full data
final_model = xgb.XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='mlogloss')
final_model.fit(X, y_encoded)

# Save the model
joblib.dump(final_model, 'model/xgboost_model.pkl')
