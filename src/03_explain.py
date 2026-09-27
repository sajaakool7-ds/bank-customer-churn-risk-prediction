import pandas as pd, numpy as np, joblib, shap, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

model = joblib.load('outputs/churn_model.joblib')
data = joblib.load('outputs/data_splits.joblib')
X_test, y_test = data['X_test'], data['y_test']

prep = model.named_steps['prep']
clf = model.named_steps['clf']

X_test_t = prep.transform(X_test)
feature_names = prep.get_feature_names_out()
X_test_df = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_df)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
print("=== Top 15 global feature importance (mean |SHAP|) ===")
print(importance.head(15))

plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
plt.tight_layout()
plt.savefig('outputs/shap_summary.png', dpi=130)
plt.close()

joblib.dump({'shap_values': shap_values, 'feature_names': feature_names, 'X_test_df': X_test_df,
             'explainer_expected_value': explainer.expected_value}, 'outputs/shap_data.joblib')

# ---- Decision system: risk tier + top reasons + recommended action ----
churn_proba = model.predict_proba(X_test)[:, 1]

def risk_tier(p):
    if p >= 0.70: return 'Critical'
    elif p >= 0.45: return 'High'
    elif p >= 0.20: return 'Medium'
    else: return 'Low'

def top_reasons(row_shap, feat_names, n=3):
    idx_sorted = np.argsort(-row_shap)
    reasons = []
    for i in idx_sorted:
        if row_shap[i] > 0:
            reasons.append(feat_names[i])
        if len(reasons) == n:
            break
    while len(reasons) < n:
        reasons.append(None)
    return reasons

def recommended_action(reasons, num_products):
    top = reasons[0] if reasons[0] else ''
    if num_products >= 3:
        return 'Immediate review by Relationship Manager'
    if 'IsActiveMember' in top or 'InactiveSenior' in top:
        return 'Re-engagement campaign'
    if 'Age' in top:
        return 'Age-specific offer and proactive call'
    if 'Germany' in top:
        return 'Review pricing and service in Germany'
    return 'Regular follow-up and general retention offer'

df_full = pd.read_csv('features.csv')
report_rows = []
customer_ids = df_full.loc[X_test.index, 'CustomerId'].values
num_products_arr = X_test['NumOfProducts'].values

for i, cust_id in enumerate(customer_ids):
    reasons = top_reasons(shap_values[i], feature_names, n=3)
    prob = churn_proba[i]
    report_rows.append({
        'CustomerId': cust_id,
        'ChurnProbability': round(float(prob), 4),
        'RiskTier': risk_tier(prob),
        'ActualExited': int(y_test.iloc[i]),
        'TopReason1': reasons[0], 'TopReason2': reasons[1], 'TopReason3': reasons[2],
        'RecommendedAction': recommended_action(reasons, num_products_arr[i])
    })

customer_risk_report = pd.DataFrame(report_rows)
customer_risk_report.to_csv('outputs/customer_risk_report.csv', index=False)
print("Saved outputs/customer_risk_report.csv:", customer_risk_report.shape)

tier_validation = customer_risk_report.groupby('RiskTier')['ActualExited'].agg(['mean', 'count']).round(3)
print(tier_validation.reindex(['Critical', 'High', 'Medium', 'Low']))
