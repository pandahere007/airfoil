
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("component_fault_data.csv")
X = df.drop("status", axis=1)
y = df["status"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Predict and save to file
df["prediction"] = model.predict(X)
df.to_csv("predicted_output.csv", index=False)

print("Predictions saved to predicted_output.csv")
