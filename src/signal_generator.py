import pandas as pd
import numpy as np
import joblib 
import os
import sys

# --- 0. CORE PATH FIX (Single, Clean) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Deep learning imports (for LSTM loading, if needed)
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler
from src.model_lstm import create_lstm_dataset  # Re-use the LSTM windowing function

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def load_data():
    """Loads the processed data for testing/prediction."""
    file_path = os.path.join(DATA_PROCESSED_DIR, 'btc_ml_features.csv')
    if not os.path.exists(file_path):
        print("❌ Error: Processed features file not found.")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path, index_col='Open Time', parse_dates=True)
    return df

def get_model_mse_loss():
    """Provides the actual MSE loss values from Day 6 and Day 7 runs."""
    # NOTE: These values must be updated based on your actual console output!
    # RFR and Prophet MSEs are large because they operate on unscaled prices (e.g., $60,000^2), 
    # while LSTM MSE is very small (operates on scaled prices 0-1).
    # For comparison, we use RFR and Prophet as "unscaled" benchmarks and LSTM as "scaled".
    
    losses = {
        # LSTM is typically scaled (very small loss)
        'LSTM': 0.0005,  # Placeholder: Replace with actual scaled MSE from Day 6
        
        # RFR and Prophet are unscaled (large loss, compared to current price squared)
        'RFR': 58481466.1866,  # Actual value from your run
        'Prophet': 26229496.1462  # Actual value from your run
    }
    return losses

def predict_rfr(df_test):
    """Loads RFR model and makes an unscaled price prediction."""
    model_path = os.path.join(MODELS_DIR, 'model_B_rfr.joblib')
    if not os.path.exists(model_path): 
        return None, "RFR Model Missing"

    model = joblib.load(model_path)
    
    # Input data for prediction is the last row of features (excluding the target)
    X_predict = df_test.drop('target', axis=1).iloc[[-1]]
    
    y_pred = model.predict(X_predict)
    
    return y_pred[0], "Price"  # Returns unscaled price prediction

def predict_prophet(df_test):
    """Loads Prophet model and makes an unscaled price prediction."""
    model_path = os.path.join(MODELS_DIR, 'model_C_prophet.joblib')
    if not os.path.exists(model_path): 
        return None, "Prophet Model Missing"

    model = joblib.load(model_path)
    
    # Prophet prediction requires defining the next timestamp (future frame)
    last_timestamp = df_test.index[-1]
    
    # Assume 4-hour interval; target is next period
    future_time = last_timestamp + pd.Timedelta(hours=4) 
    future = pd.DataFrame({'ds': [future_time]})
    
    forecast = model.predict(future)
    
    # Prophet's forecast is stored in the 'yhat' column
    return forecast['yhat'].iloc[0], "Price" 

def predict_lstm(df_test):
    """Loads LSTM model and makes a scaled price prediction (then inverse scales)."""
    model_path = os.path.join(MODELS_DIR, 'model_A_lstm.h5')
    scaler_path = os.path.join(MODELS_DIR, 'scaler_lstm.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, "LSTM Model/Scaler Missing"
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare windowed data (reuse from training)
    window_size = 60  # Adjust if different from training
    X_last = create_lstm_dataset(df_test.drop('target', axis=1), window_size, train=False)[-1:]  # Last window
    
    # Scale and predict
    X_scaled = scaler.transform(X_last.reshape(X_last.shape[0], -1))
    pred_scaled = model.predict(X_scaled)
    pred_price = scaler.inverse_transform(np.hstack([pred_scaled, np.zeros((1, X_last.shape[1] - pred_scaled.shape[1]))]))[0, 0]
    
    return pred_price, "Price (Scaled & Inversed)"

# --- 4. Model Comparison and Selection ---

def model_comparison():
    """Compares all models and selects the most accurate one (for simplicity, we choose RFR)."""
    
    losses = get_model_mse_loss()
    
    # In practice, comparing scaled (LSTM) vs. unscaled (RFR/Prophet) loss is complicated.
    # Since RFR showed the lowest absolute unscaled loss among RFR/Prophet, 
    # and is simplest for price prediction, we select it as the default "Best Model" for signals.
    
    # Find the minimum loss among RFR and Prophet (since they are comparable unscaled)
    unscaled_losses = {k: v for k, v in losses.items() if k != 'LSTM'}
    best_unscaled_model = min(unscaled_losses, key=unscaled_losses.get)
    
    return best_unscaled_model, losses

# --- 5. Automated Trading Signal Generation ---

def generate_trading_signal(prediction, current_price, features):
    """Generates a Buy/Sell/Hold signal and Risk level."""
    
    # A. Prediction Direction (0.3% threshold)
    prediction_change = (prediction - current_price) / current_price
    
    # B. Sentiment/Indicator Confirmation
    last_features = features.iloc[-1]
    
    # Check for the correct MACD column name (try common variants)
    possible_macd_cols = ['MACDh_12_26_9', 'MACD_Histogram', 'MACDH']
    macd_col = next((col for col in possible_macd_cols if col in last_features.index), None)
    
    if macd_col:
        macd_positive = last_features[macd_col] > 0
    else:
        print(f"⚠️ MACD Key Error: No MACD column found in {possible_macd_cols}. MACD confirmation skipped.")
        macd_positive = True  # Neutral fallback
    
    # Bullish Confirmation: MACD Histogram positive AND News Sentiment positive (> 0.1)
    is_bullish_confirmed = macd_positive and (last_features['News_Sentiment'] > 0.1)
    
    # Bearish Confirmation: RSI >70 (Overbought) OR F&G >=75 (Extreme Greed) 
    is_bearish_confirmed = (last_features['RSI_14'] > 70) or (last_features['FNG_Index'] >= 75) 
    
    # C. Determine Signal
    if prediction_change > 0.003 and is_bullish_confirmed:
        signal = 'BUY'
    elif prediction_change < -0.003 and is_bearish_confirmed:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    # D. Determine Risk Level (Simple: Based on recent volatility proxy from Close std dev)
    recent_vol = features['Close'].tail(10).std() / current_price  # 10-period normalized volatility
    if recent_vol < 0.02:
        risk = 'Low'
    elif recent_vol < 0.05:
        risk = 'Medium'
    else:
        risk = 'High'
    
    print(f"🔍 Debug: Pred Change: {prediction_change:.2%}, MACD Pos: {macd_positive}, Bull Conf: {is_bullish_confirmed}, Bear Conf: {is_bearish_confirmed}, Vol: {recent_vol:.2%}")
    
    return signal, prediction_change, risk

# --- 6. Main Runner ---

def run_signal_generator():
    """Executes the full model comparison and signal generation pipeline."""
    df = load_data()
    
    if df.empty:
        print("Cannot run signal generator: Data is empty.")
        return

    # A. Model Comparison
    best_model, losses = model_comparison()
    print("\n--- Model Comparison ---")
    for model_name, loss in losses.items():
        print(f"Model Performance: {model_name} Loss: {loss:.4f}")
    print(f"-> Used for real-time prediction: {best_model}")
    print("------------------------")
    
    # B. Get Prediction from Best Model
    if best_model == 'RFR':
        prediction, prediction_type = predict_rfr(df)
    elif best_model == 'Prophet':
        prediction, prediction_type = predict_prophet(df)
    elif best_model == 'LSTM':
        prediction, prediction_type = predict_lstm(df)
    else:  # Fallback
        prediction, prediction_type = predict_rfr(df)
        best_model = 'RFR (Fallback)'
        
    if prediction is None:
        print(f"❌ Fatal: {best_model} model prediction failed. Cannot generate signal.")
        return
    
    last_close_price = df['Close'].iloc[-1]
        
    # C. Generate Signal
    signal, change_percent, risk = generate_trading_signal(prediction, last_close_price, df.drop('target', axis=1))

    # D. Output
    print("\n--- Automated Trading Signal ---")
    print(f"Model Used: {best_model}")
    print(f"Current Price (Close): ${last_close_price:,.2f}")
    print(f"Predicted Next Price ({prediction_type}): ${prediction:,.2f}")
    print(f"Predicted Change: {change_percent * 100:.2f}%")
    print(f"Signal: {signal}")
    print(f"Risk Level: {risk}")
    print("------------------------------")


if __name__ == '__main__':
    run_signal_generator()