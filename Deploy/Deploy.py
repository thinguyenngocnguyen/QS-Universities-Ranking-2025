# StudentC_app.py
# Track C: The Deployer Student - Khush Barot / GROUP 1
# This script trains the model, saves it, and runs a live prediction on new data.

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split


# PART 1 Loading  and cleaning the data

df = pd.read_csv("qs-world-rankings-2025.csv")

new_df = df.copy()

new_df['2025 Rank'] = pd.to_numeric(new_df['2025 Rank'].astype(str).str.replace('=', '', regex=False), errors='coerce')
new_df['2024 Rank'] = pd.to_numeric(new_df['2024 Rank'].astype(str).str.replace('=', '', regex=False), errors='coerce')

# cleaning up overall score column
new_df['QS Overall Score'] = new_df['QS Overall Score'].replace('-', np.nan)
new_df['QS Overall Score'] = pd.to_numeric(new_df['QS Overall Score'], errors='coerce')

# filling missing values with median 
numeric_cols = new_df.select_dtypes(include=[np.number]).columns
new_df[numeric_cols] = new_df[numeric_cols].fillna(new_df[numeric_cols].median())


# Part 2 feature engineering
new_df['Rank Movement'] = new_df['2024 Rank'] - new_df['2025 Rank']
new_df['Reputation Score'] = new_df[['Academic Reputation', 'Employer Reputation']].mean(axis=1)
new_df['Research Strength'] = new_df[['Citations per Faculty', 'International Research Network']].mean(axis=1)
new_df['Internationalization Score'] = new_df[['International Faculty', 'International Students', 'International Research Network']].mean(axis=1)
new_df['Career Outcomes Score'] = new_df[['Employer Reputation', 'Employment Outcomes']].mean(axis=1)
new_df['Student Support Score'] = new_df[['Faculty Student', 'International Students']].mean(axis=1)
new_df['International Balance Gap'] = (new_df['International Faculty'] - new_df['International Students']).abs()
new_df['Reputation Gap'] = new_df['Academic Reputation'] - new_df['Employer Reputation']

# target variable: 1 if university is ranked top 50, 0 if not
new_df['Top 50 Flag'] = (new_df['2025 Rank'] <= 50).astype(int)


# PART 3 training the model to save it 

# 7 features to use for prediction 
features = [
    'Reputation Score',
    'Research Strength',
    'Internationalization Score',
    'Career Outcomes Score',
    'Student Support Score',
    'International Balance Gap',
    'Reputation Gap'
]

X = new_df[features]
y = new_df['Top 50 Flag']

# split into train and test sets (same split as group notebook)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaling the data, fits only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training the gradient boosting model (same settings as group notebook)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# saving the model and scaler to disk so we can reuse them
joblib.dump(model, "saved_model.joblib")
joblib.dump(scaler, "saved_scaler.joblib")
print("Model and scaler saved!")


# Part 4 loading the same model and running a live prediction
# load them back from disk
model = joblib.load("saved_model.joblib")
scaler = joblib.load("saved_scaler.joblib")
print("Model and scaler loaded!")


#  PART 5: Live demo input a new university and get a prediction

academic_reputation = 95.0
employer_reputation = 92.0
citations_per_faculty = 99.0
international_research_network = 88.0
international_faculty = 72.0
international_students = 68.0
employment_outcomes = 90.0
faculty_student = 85.0


reputation_score = (academic_reputation + employer_reputation) / 2
research_strength = (citations_per_faculty + international_research_network) / 2
internationalization_score = (international_faculty + international_students + international_research_network) / 3
career_outcomes_score = (employer_reputation + employment_outcomes) / 2
student_support_score = (faculty_student + international_students) / 2
international_balance_gap = abs(international_faculty - international_students)
reputation_gap = academic_reputation - employer_reputation


new_university = pd.DataFrame([[
    reputation_score,
    research_strength,
    internationalization_score,
    career_outcomes_score,
    student_support_score,
    international_balance_gap,
    reputation_gap
]], columns=features)

new_university_scaled = scaler.transform(new_university)

# prediction
prediction = model.predict(new_university_scaled)[0]
probability = model.predict_proba(new_university_scaled)[0]

# print the results
print("\n----------------------------------------")
print("         LIVE DEMO - TOP 50 PREDICTOR")
print("\nInput scores entered:")
print(f"  Academic Reputation:            {academic_reputation}")
print(f"  Employer Reputation:            {employer_reputation}")
print(f"  Citations per Faculty:          {citations_per_faculty}")
print(f"  International Research Network: {international_research_network}")
print(f"  International Faculty:          {international_faculty}")
print(f"  International Students:         {international_students}")
print(f"  Employment Outcomes:            {employment_outcomes}")
print(f"  Faculty Student:                {faculty_student}")

print("\nEngineered features used by model:")
print(f"  Reputation Score:               {round(reputation_score, 2)}")
print(f"  Research Strength:              {round(research_strength, 2)}")
print(f"  Internationalization Score:     {round(internationalization_score, 2)}")
print(f"  Career Outcomes Score:          {round(career_outcomes_score, 2)}")
print(f"  Student Support Score:          {round(student_support_score, 2)}")
print(f"  International Balance Gap:      {round(international_balance_gap, 2)}")
print(f"  Reputation Gap:                 {round(reputation_gap, 2)}")

if prediction == 1:
    print(f"\nPREDICTION  -->  TOP 50 ")
else:
    print(f"\nPREDICTION  -->  NOT Top 50 ")

print(f"Confidence  -->  Top 50: {round(probability[1]*100, 2)}%  |  Not Top 50: {round(probability[0]*100, 2)}%")
print("----------------------------------------")