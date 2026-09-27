import pandas as pd

df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(columns=['RowNumber', 'Surname'])  # CustomerId يضل للتقرير النهائي

# ---- Feature engineering ----
df['ZeroBalance'] = (df['Balance'] == 0).astype(int)
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['HighProductCount'] = (df['NumOfProducts'] >= 3).astype(int)   # very strong churn signal
df['IsGermany'] = (df['Geography'] == 'Germany').astype(int)
df['TenureByAge'] = df['Tenure'] / df['Age']
df['CreditScoreBin'] = pd.cut(df['CreditScore'], bins=[0, 580, 670, 740, 800, 850],
                               labels=['Poor', 'Fair', 'Good', 'VeryGood', 'Excellent'])
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 30, 40, 50, 60, 100],
                         labels=['18-30', '31-40', '41-50', '51-60', '60+'])
df['InactiveSenior'] = ((df['IsActiveMember'] == 0) & (df['Age'] >= 50)).astype(int)
df['ProductsPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)

df.to_csv('features.csv', index=False)
print(df.shape)
print(df.columns.tolist())
