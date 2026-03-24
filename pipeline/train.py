import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("crm_data.csv")

# -----------------------------
# 2. Prepare data
# -----------------------------
X = df[["clicks", "time_spent", "email_opened"]]
y = df["converted"]

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate model
# -----------------------------
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# -----------------------------
# 6. Save model (FIXED)
# -----------------------------
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))

print("✅ Model saved successfully!")