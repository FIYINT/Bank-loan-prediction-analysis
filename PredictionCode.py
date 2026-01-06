import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
df = pd.read_csv(r'Downloads/microfinance_loan_prediction_dataset.csv')

# 2. PRE-PROCESSING (Internal only, for the AI to work)
le_gender = LabelEncoder()
le_occ = LabelEncoder()
df['Gender_Enc'] = le_gender.fit_transform(df['Gender'])
df['Occupation_Enc'] = le_occ.fit_transform(df['Occupation'])

features = ['Age', 'Gender_Enc', 'Occupation_Enc', 'Monthly_Income', 
            'Loan_Amount', 'Repayment_Term_Months', 'Past_Defaults', 'Existing_Loans']

# 3. GENERATE PREDICTIONS (Default & Scoring)
X = df[features]
y_class = df['Loan_Paid'].map({'Yes': 1, 'No': 0})

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y_class)

# 4. GENERATE PREDICTIONS (Timing/Days)
# Train only on those who paid, then predict for everyone
train_reg = df[df['Loan_Paid'] == 'Yes'].dropna(subset=['Days_to_Repay'])
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(train_reg[features], train_reg['Days_to_Repay'])

# 5. CREATE THE "RESULTS ONLY" DATAFRAME
# We only grab the ID and the new AI insights
ai_results = pd.DataFrame()
ai_results['Customer_ID'] = df['Customer_ID']
ai_results['AI_Repayment_Score'] = (clf.predict_proba(X)[:, 1] * 100).round(2)
ai_results['AI_Status_Prediction'] = clf.predict(X)
ai_results['AI_Status_Prediction'] = ai_results['AI_Status_Prediction'].map({1: 'Likely to Repay', 0: 'Likely to Default'})
ai_results['AI_Expected_Days'] = reg.predict(X).astype(int)

# 6. EXPORT TO NEW CSV
ai_results.to_csv('Predictive_Results.csv', index=False)

print("AI results exported to 'Predictive_Results.csv'.")