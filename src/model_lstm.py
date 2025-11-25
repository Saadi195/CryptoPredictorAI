import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import sys

# --- 0. PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --- 1. Data Preparation for LSTM (Scaling and Windowing) ---

def create_lstm_dataset(features, target, time_steps=1):
    """
    Transforms time-series data into windows (sequences) suitable for LSTM.
    
    :param features: 2D array of input features (X).
    :param target: 1D array of target values (y).
    :param time_steps: The lookback window size (sequence length).
    :return: 3D features array (samples, time_steps, features) and 2D target array.
    """
    X, y = [], []
    for i in range(len(features) - time_steps):
        # Create a sequence of 'time_steps' length for the current sample
        X.append(features[i:(i + time_steps), :])
        # The target is the value immediately after the sequence
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)


def preprocess_data(time_steps=8):
    """Loads, splits, and scales data for LSTM training."""
    file_path = os.path.join(DATA_PROCESSED_DIR, 'btc_ml_features.csv')
    print(f"Loading features from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Features file not found. Run Day 5 first: {file_path}")

    df = pd.read_csv(file_path, index_col='Open Time', parse_dates=True)

    # Separate features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Scale data to be between 0 and 1 (REQUIRED for deep learning)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    # We fit/transform the target separately but use it to reconstruct predictions later
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Create time-series windows (Time Step = 8 corresponds to 8 * 4h = 32 hours lookback)
    X_windowed, y_windowed = create_lstm_dataset(X_scaled, y_scaled, time_steps)
    
    # Define split index (e.g., use the last 10% of data for validation/testing)
    split_index = int(0.9 * len(X_windowed))
    
    X_train, X_test = X_windowed[:split_index], X_windowed[split_index:]
    y_train, y_test = y_windowed[:split_index], y_windowed[split_index:]

    print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")
    return X_train, y_train, X_test, y_test, scaler_y, time_steps

# --- 2. Build and Train LSTM Model ---

def build_lstm_model(input_shape):
    """Defines the LSTM architecture."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1) # Output layer: predicts one value (the future price)
    ])
    
    # Using Adam optimizer and Mean Squared Error (MSE) for regression
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate_lstm():
    """Main function to run the training pipeline."""
    try:
        X_train, y_train, X_test, y_test, scaler_y, time_steps = preprocess_data()
    except FileNotFoundError as e:
        print(e)
        return None, None

    # Define input shape: (time_steps, number_of_features)
    input_shape = (time_steps, X_train.shape[2])
    
    # Build and train
    model = build_lstm_model(input_shape)
    print(f"\nTraining LSTM model with input shape: {input_shape}")
    
    # Fit the model (epochs = number of passes through the data)
    history = model.fit(
        X_train, 
        y_train, 
        epochs=50, # Adjust epochs based on runtime and performance
        batch_size=32, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # --- 3. Save Model and Evaluate ---
    
    # Save the trained model
    model_path = os.path.join(MODELS_DIR, 'model_A_lstm.h5')
    model.save(model_path)
    print(f"\n✅ Model A (LSTM) saved to: {model_path}")
    
    # Predict and inverse transform to get actual prices
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_y.inverse_transform(y_test)
    
    # Calculate simple direction accuracy (Did it predict UP or DOWN correctly?)
    # Direction is determined by comparing the target (future price) to the last known close price (Close column)
    
    # To properly calculate accuracy, we need the original closing prices, which is complex here.
    # For a simple metric, let's use the directional accuracy of the *change*.
    
    # Calculate the change: (Prediction - Test) / Test
    # For demonstration, we'll focus on the loss metric (MSE) for performance display in Day 10.
    mse_loss = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"LSTM Model Test MSE Loss (Scaled): {mse_loss:.6f}")
    
    return model, mse_loss

if __name__ == '__main__':
    train_and_evaluate_lstm()