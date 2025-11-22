import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Pth setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(project_root, "Dataset_Generation", "micro_business_dataset_500.csv")
models_path = os.path.join(project_root, "Models_Files")
os.makedirs(models_path, exist_ok=True)

# Load dataset
df = pd.read_csv(csv_path)

# Encode categorical columns
categorical_cols = ["Business", "City", "Product/Service", "Marketing_Channel"]
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_cols])

# Numeric features
numeric_data = df[["Startup_Cost_PKR", "Cost_per_Unit", "Price_per_Unit"]].values

# Combine features
X = np.hstack((encoded_data, numeric_data))

# Targets
y_profit = (df["Price_per_Unit"] - df["Cost_per_Unit"]) * 30  # monthly estimate
y_failure = df["Failure_Risk"]

# Train models
profit_model = RandomForestRegressor(n_estimators=100, random_state=42)
profit_model.fit(X, y_profit)

failure_model = RandomForestRegressor(n_estimators=100, random_state=42)
failure_model.fit(X, y_failure)

# Save models
joblib.dump(profit_model, os.path.join(models_path, "profit_model.pkl"))
joblib.dump(failure_model, os.path.join(models_path, "failure_model.pkl"))
joblib.dump(encoder, os.path.join(models_path, "encoder.pkl"))

print("Models trained and saved successfully!")
