# train_model.py
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Dummy dataset: 8 features (Yes/No mapped to 1/0, plus contract numeric)
X = np.array([
    [1,1,0,1,1,0,1,1],
    [0,0,0,1,0,0,0,2],
    [1,0,1,1,1,1,0,1],
    [0,1,0,0,0,0,1,2],
    [1,1,1,1,1,1,1,1]
])

y = np.array([1, 0, 1, 0, 1])  # 1 = churn, 0 = not churn

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as churn_model.pkl")
