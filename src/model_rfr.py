import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib # Used for saving scikit-learn models
import os
import sys

# --- 0. PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def train_and_evaluate_rfr():
    """Loads data, trains Random Forest Regressor, and saves the model."""
    file_path = os.path.join(DATA_PROCESSED_DIR, 'btc_ml_features.csv')
    print(f"Loading features for Random Forest from: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ Error: ML features file not found. Skipping RFR.")
        return

    df = pd.read_csv(file_path, index_col='Open Time', parse_dates=True)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # --- 1. Split Data ---
    # Use 90% for training, 10% for testing (consistent with LSTM)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )
    
    print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")
    
    # --- 2. Build and Train RFR Model ---
    print("\nTraining Model B: Random Forest Regressor...")
    
    # Initialize RFR (using a modest number of estimators for speed)
    rfr_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1 # Use all available cores
    )
    
    rfr_model.fit(X_train, y_train)
    
    # --- 3. Save Model and Evaluate ---
    
    # Save the trained model using joblib (standard for scikit-learn models)
    model_path = os.path.join(MODELS_DIR, 'model_B_rfr.joblib')
    joblib.dump(rfr_model, model_path)
    print(f"\n✅ Model B (Random Forest) saved to: {model_path}")
    
    # Predict and evaluate
    y_pred = rfr_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RFR Model Test MSE Loss: {mse:.4f}")
    print(f"RFR Model Test R-squared: {r2:.4f}")
    
    return mse

if __name__ == '__main__':
    train_and_evaluate_rfr()