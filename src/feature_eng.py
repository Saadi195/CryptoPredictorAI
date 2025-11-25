import pandas as pd
import pandas_ta as ta
import os
import sys

# --- 0. PATH FIX AND GLOBAL DEFINITIONS ---
# Add project root to path for absolute directory resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def load_and_preprocess_market_data(file_path):
    """Loads and preprocesses market (OHLCV) data."""
    print(f"Loading market data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ Error: Market data file not found.")
        return pd.DataFrame()
        
    df = pd.read_csv(file_path, index_col='Open Time', parse_dates=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.astype(float)
    return df

def generate_technical_features(df):
    """Calculates key technical indicators (features)."""
    if df.empty:
        return df

    print("🛠️ Generating Technical Features (Indicators)...")

    # 1. Moving Averages (SMA/EMA)
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=50, append=True)

    # 2. Momentum Indicators
    df.ta.rsi(length=14, append=True) 
    df.ta.macd(append=True) # Generates MACD, MACDH (Histogram), MACDS (Signal)

    # 3. Volatility Indicators
    df.ta.bbands(append=True) # Generates BBL, BBM, BBU, etc.

    # 4. Volume Indicators
    df.ta.obv(append=True)

    return df

def run_feature_engineering():
    """Merges all data streams, creates features, and defines the target."""
    
    # 1. Load Data
    market_data_path = os.path.join(DATA_RAW_DIR, 'btc_historical_4h.csv')
    df_market = load_and_preprocess_market_data(market_data_path)
    
    if df_market.empty:
        print("Cannot proceed with feature engineering. Market data is empty.")
        return

    # 2. Generate Features on Market Data
    df_features = generate_technical_features(df_market)
    
    # 3. Define the Target Variable
    # Predict the Close price 1 step (4 hours) into the future.
    df_features['target'] = df_features['Close'].shift(-1)
    
    # 4. Load & Add F&G Index (Dynamic Feature)
    fng_path = os.path.join(DATA_RAW_DIR, 'crypto_fng_index.csv')
    try:
        df_fng = pd.read_csv(fng_path)
        # Use the most recent F&G Index value (e.g., 20)
        fng_value = df_fng['fng_index'].iloc[-1]
        df_features['FNG_Index'] = fng_value
        print(f"✅ F&G Index loaded and added: {fng_value}")
    except Exception:
        # Default to neutral if the file is missing or contains error data
        df_features['FNG_Index'] = 50.0 
        print("⚠️ F&G Index load failed. Defaulting to 50 (Neutral).")


    # 5. Add NLP Sentiment Score (Placeholder for Day 5)
    # We will compute the real sentiment tomorrow, but for now, we use a placeholder score.
    df_features['News_Sentiment'] = 0.5 # Placeholder: Neutral score
    print("⚠️ News_Sentiment added as placeholder. Will be computed on Day 5.")

    # 6. Final Cleanup
    initial_rows = len(df_features)
    # Drop rows with NaN: These are the rows at the start (needed by indicators) 
    # and the very last row (where 'target' is NaN).
    df_final = df_features.dropna()
    
    print(f"   Dropped {initial_rows - len(df_final)} rows with NaN values.")

    # 7. Save Processed Data
    if not os.path.exists(DATA_PROCESSED_DIR):
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        
    output_path = os.path.join(DATA_PROCESSED_DIR, 'btc_ml_features.csv')
    df_final.to_csv(output_path)
    
    print(f"\n✅ Feature Engineering Complete!")
    print(f"   Final ML-ready data saved to: {output_path}")
    print(f"   Final DataFrame shape: {df_final.shape}")
    print(df_final.tail())
    

if __name__ == '__main__':
    run_feature_engineering()