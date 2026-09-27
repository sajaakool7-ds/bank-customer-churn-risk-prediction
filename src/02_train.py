import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)
df = pd.read_csv('features.csv')

target = 'Exited'
id_col = 'CustomerId'
X = df.drop(columns=[id_col, target])
y = df[target]

cat_cols = ['Geography', 'Gender', 'CreditScoreBin', 'AgeGroup']
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), cat_cols)
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Logistic Regression (baseline) ----
logreg_pipe = Pipeline([('prep', preprocess),
                         ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))])
logreg_pipe.fit(X_train, y_train)
logreg_proba = logreg_pipe.predict_proba(X_test)[:, 1]
print("LogReg ROC-AUC:", round(roc_auc_score(y_test, logreg_proba), 4))

# ---- Random Forest (tuned) ----
rf_pipe = Pipeline([('prep', preprocess),
                     ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])
rf_params = {
    'clf__n_estimators': [200, 400, 600],
    'clf__max_depth': [4, 6, 8, 12, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2']
}
rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=25, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
rf_proba = rf_best.predict_proba(X_test)[:, 1]
print("RF best params:", rf_search.best_params_)
print("RF ROC-AUC:", round(roc_auc_score(y_test, rf_proba), 4))

# ---- XGBoost (tuned, scale_pos_weight) ----
neg, pos = np.bincount(y_train)
spw = neg / pos
xgb_pipe = Pipeline([('prep', preprocess),
                      ('clf', xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                                 scale_pos_weight=spw, random_state=42, n_jobs=-1,
                                                 tree_method='hist'))])
xgb_params = {
    'clf__n_estimators': [200, 300, 500],
    'clf__max_depth': [3, 4, 5, 6],
    'clf__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'clf__subsample': [0.7, 0.8, 1.0],
    'clf__colsample_bytree': [0.6, 0.8, 1.0],
    'clf__min_child_weight': [1, 3, 5]
}
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_params, n_iter=30, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
xgb_proba = xgb_best.predict_proba(X_test)[:, 1]
print("XGB best params:", xgb_search.best_params_)
print("XGB ROC-AUC:", round(roc_auc_score(y_test, xgb_proba), 4))

# ---- XGBoost + SMOTE ----
xgb_smote_pipe = ImbPipeline([('prep', preprocess), ('smote', SMOTE(random_state=42)),
                               ('clf', xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                                          random_state=42, n_jobs=-1,
                                                          n_estimators=300, max_depth=4, learning_rate=0.05))])
xgb_smote_pipe.fit(X_train, y_train)
smote_proba = xgb_smote_pipe.predict_proba(X_test)[:, 1]
print("XGB+SMOTE ROC-AUC:", round(roc_auc_score(y_test, smote_proba), 4))

joblib.dump(xgb_best, 'outputs/churn_model.joblib')
joblib.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
             'cat_cols': cat_cols, 'num_cols': num_cols}, 'outputs/data_splits.joblib')
print("Saved outputs/churn_model.joblib")
