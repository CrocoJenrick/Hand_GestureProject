import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

if not os.path.exists("gesture_dataset.csv"):
    print("Error: gesture_dataset.csv not found.")
    exit()

print("Loading data...")
df = pd.read_csv("gesture_dataset.csv")
if df.empty:
    print("Error: Dataset is empty.")
    exit()

X = df.drop("label", axis=1)
y = df["label"]

print(f"Training on {len(df)} samples...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Success: Model saved as gesture_model.pkl!")
