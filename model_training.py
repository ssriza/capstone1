# model_training.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("telecom_churn_data.csv")

# 2. Remove duplicates & fill missing
df = df.drop_duplicates()
df = df.fillna("No")  # Fill missing with "No" for categoricals

# 3. Keep only the 8 features + target
selected_features = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'OnlineSecurity', 'TechSupport', 'Contract', 'MonthlyCharges'
]
target_column = 'Churn'
df = df[selected_features + [target_column]]

# 4. Store label encoders for later
label_encoders = {}

# 5. Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 6. Split into X and y
X = df[selected_features]
y = df[target_column]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 8. Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# 11. Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "features.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n Model, scaler, features, and label encoders saved!")
