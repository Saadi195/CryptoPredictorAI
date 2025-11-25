# In src/model_prophet.py, near the top
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics
import joblib 
import os
import sys

# 🚨 ADD THIS IMPORT LINE 🚨
from sklearn.metrics import mean_squared_error 
# -----------------------------

# --- 0. PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def train_and_evaluate_prophet():
    """Loads price data, trains Prophet model, and evaluates performance."""
    file_path = os.path.join(DATA_PROCESSED_DIR, 'btc_ml_features.csv')
    print(f"Loading data for Prophet from: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ Error: ML features file not found. Skipping Prophet.")
        return

    df = pd.read_csv(file_path, index_col='Open Time', parse_dates=True)

    # --- 1. Prepare Data for Prophet ---
    # Prophet requires columns named 'ds' (datestamp) and 'y' (target value)
    prophet_df = df.reset_index()[['Open Time', 'target']].rename(
        columns={'Open Time': 'ds', 'target': 'y'}
    )
    
    # Define split index (consistent 90% training)
    split_index = int(0.9 * len(prophet_df))
    train_df = prophet_df[:split_index]
    test_df = prophet_df[split_index:]
    
    print(f"Data split: Train samples={len(train_df)}, Test samples={len(test_df)}")

    # --- 2. Build and Train Prophet Model ---
    print("\nTraining Model C: Prophet...")
    
    # Initialize Prophet (default settings)
    model = Prophet(
        yearly_seasonality=False, # Crypto data is usually non-yearly cyclical
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.95 # Confidence interval
    )
    
    model.fit(train_df)
    
    # --- 3. Save Model and Evaluate ---
    
    # Save the trained model using joblib (or Prophet's internal serialization)
    model_path = os.path.join(MODELS_DIR, 'model_C_prophet.joblib')
    # joblib is used here as a clean wrapper for saving the Prophet object
    joblib.dump(model, model_path)
    print(f"\n✅ Model C (Prophet) saved to: {model_path}")
    
    # Create future dataframe for evaluation period
    future = test_df.drop('y', axis=1) 
    forecast = model.predict(future)
    
    # Simple MSE calculation for comparison (comparing forecasted yhat to actual y)
    prophet_mse = mean_squared_error(test_df['y'], forecast['yhat'][:len(test_df)])
    
    print(f"Prophet Model Test MSE Loss: {prophet_mse:.4f}")
    
    return prophet_mse

if __name__ == '__main__':
    train_and_evaluate_prophet()