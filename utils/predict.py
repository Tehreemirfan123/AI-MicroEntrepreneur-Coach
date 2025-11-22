import os
import joblib
import numpy as np

# Get absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models using absolute paths
profit_model_path = os.path.join(PROJECT_ROOT, "Models_Files", "profit_model.pkl")
failure_model_path = os.path.join(PROJECT_ROOT, "Models_Files", "failure_model.pkl")
encoder_path = os.path.join(PROJECT_ROOT, "Models_Files", "encoder.pkl")

profit_model = joblib.load(profit_model_path)
failure_model = joblib.load(failure_model_path)
encoder = joblib.load(encoder_path)

# Prepare input
def prepare_input(sample, encoder):
    import pandas as pd
    import numpy as np

    # Create dataframe with zeros for all encoded features
    input_df = pd.DataFrame(columns=encoder.get_feature_names_out())
    input_df.loc[0] = 0

    # Numeric features
    numeric_cols = ["Startup_Cost_PKR", "Cost_per_Unit", "Price_per_Unit"]
    for col in numeric_cols:
        input_df[col] = sample[col]

    # Encode categorical features
    cat_features = encoder.transform([[sample["Business"], sample["City"], sample["Product/Service"], sample["Marketing_Channel"]]])
    cat_cols = encoder.get_feature_names_out()
    for i, col in enumerate(cat_cols):
        input_df[col] = cat_features[0][i]

    return input_df

def predict_profit(sample, profit_model, encoder):
    input_df = prepare_input(sample, encoder)
    return profit_model.predict(input_df)[0]

def predict_failure(sample, failure_model, encoder):
    input_df = prepare_input(sample, encoder)
    return failure_model.predict(input_df)[0]
