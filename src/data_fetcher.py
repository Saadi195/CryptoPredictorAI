
    # --- At the very top of src/data_fetcher.py ---
import os
import sys
# --- Path Fix for Configuration File ---
# Add the project root to the path so config.py can be imported from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------------------

import pandas as pd
import requests
from bs4 import BeautifulSoup
from binance.client import Client
# ... (other imports) ...


# --- 1. CONFIGURATION LOADING (This remains the same, but now it should work) ---
try:
    # This import should now succeed because the path is set above
    from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, CRYPTO_NEWS_URL
    
    # Rest of the config loading logic...
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    print("Binance Client Initialized with API Keys.")
except ImportError:
    # This block now means EITHER the file is missing/empty, OR the variables aren't defined.
    print("Warning: Could not import config variables. Using public client and skipping news scrape.")
    client = Client()
    # Define placeholder values to avoid NameErrors later in the script
    BINANCE_API_KEY, BINANCE_SECRET_KEY, CRYPTO_NEWS_URL = None, None, None 
except Exception as e:
    print(f"Error during config initialization: {e}. Using public client.")
    client = Client()
    BINANCE_API_KEY, BINANCE_SECRET_KEY, CRYPTO_NEWS_URL = None, None, None 

# ... (The rest of your file content remains the same, but the final part needs adjusting)
# 🚨 Path Fix: Define the absolute base directory 🚨
# This ensures all saves occur relative to the script's location,
# then moves up one level (..) to the main project directory, 
# and then into data/raw.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of data_fetcher.py (src)
PROJECT_ROOT = os.path.join(BASE_DIR, '..')           # CryptoPredictorAI folder
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw') # Final target folder
# --- 1. CONFIGURATION LOADING ---
# Safely try to import credentials, falling back to a public client if keys are missing.
try:
    from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, CRYPTO_NEWS_URL
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    print("Binance Client Initialized with API Keys.")
except ImportError:
    print("Warning: Could not import API keys from config.py. Using public Binance client.")
    # Fallback to public client for market data endpoints
    client = Client()
except Exception as e:
    print(f"Error initializing Binance client: {e}. Using public client.")
    client = Client()

# --- 2. MARKET DATA FETCHER (API) ---

def fetch_historical_klines(symbol='BTCUSDT', interval='4h', start_str='90 days ago'):
    """
    Fetches historical candlestick (OHLCV) data from Binance and returns a DataFrame.
    
    :param symbol: Trading pair (e.g., 'BTCUSDT').
    :param interval: Timeframe ('1h', '4h', '1d', etc.).
    :param start_str: Starting point (e.g., '1 Jan, 2024' or '30 days ago').
    :return: Pandas DataFrame with datetime index and float OHLCV columns.
    """
    print(f"\n📈 Fetching {symbol} data ({interval}) starting {start_str}...")
    
    try:
        # get_historical_klines automatically handles fetching more than 1000 data points.
        klines = client.get_historical_klines(symbol, interval, start_str)
    except Exception as e:
        print(f"❌ Error fetching Binance data: {e}")
        return pd.DataFrame()

    # Define the column names for the 12 columns returned by the Binance API
    columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ]
    
    data = pd.DataFrame(klines, columns=columns)
    
    if data.empty:
        print("No data returned from API.")
        return data
        
    # Clean and format the DataFrame
    data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
    data.set_index('Open Time', inplace=True)
    
    # Select key columns and convert them to numeric types
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    print(f"✅ Successfully fetched {len(data)} data points.")
    return data

# --- 3. STATIC NEWS SCRAPER (BEAUTIFULSOUP) ---

def scrape_crypto_news(url):
    """
    Scrapes static news headlines and links from a given URL using requests and BeautifulSoup.
    
    :param url: The target URL for scraping.
    :return: List of dictionaries with article details.
    """
    print(f"\n📰 Scraping news from: {url}...")
    articles = []
    
    # 1. Send an HTTP GET request
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching URL {url}: {e}")
        return articles

    # 2. Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # --- ADJUST THESE SELECTORS BASED ON YOUR CHOSEN NEWS SITE ---
    # This example uses selectors typical for many news sites (you might need developer tools to adjust these)
    
    # Try to find common article wrappers
    news_items = soup.find_all(['article', 'div'], class_=lambda x: x and ('post-card' in x or 'article-item' in x or 'news-item' in x))
    
    if not news_items:
        print("Warning: No clear news items found. Check selectors.")
        # Attempt a broader search for all links with titles
        news_items = soup.find_all('a', title=True)

    for item in news_items[:20]: # Limit to 20 for quick testing
        headline, link = None, None
        
        # Scenario 1: Item is an <article> or <div> wrapper
        headline_tag = item.find(['h2', 'h3', 'a'], class_=lambda x: x and ('title' in x or 'headline' in x))
        if headline_tag:
            headline = headline_tag.get_text(strip=True)
            link = headline_tag.get('href') if headline_tag.name == 'a' else item.find('a').get('href')
        
        # Scenario 2: Item is directly an <a> tag (from the broader search)
        elif item.name == 'a' and item.get('title'):
            headline = item['title'].strip()
            link = item.get('href')
            
        if headline and link:
            # Ensure the link is absolute
            if not link.startswith('http'):
                link = url.split('/')[0] + '//' + url.split('/')[2] + link

            articles.append({
                'timestamp': pd.Timestamp.now(),
                'headline': headline,
                'link': link,
                'source': url
            })
            
    print(f"✅ Scraped {len(articles)} headlines.")
    return articles

# --- 5. EXECUTION AND SAVING (UPDATED FOR ABSOLUTE PATH) ---

# --- 5. EXECUTION AND SAVING (CLEANED AND CONSOLIDATED) ---

def run_data_fetchers():
    """Main function to run all fetchers and save the data."""
    
    # 🚨 STEP 1: DEFINE PATHS AND CREATE DIRECTORY 🚨
    # Define file paths using the absolute directory (CRUCIAL)
    price_file_path = os.path.join(DATA_RAW_DIR, 'btc_historical_4h.csv')
    news_file_path = os.path.join(DATA_RAW_DIR, 'crypto_news_static.csv')
    
    # Robust Directory Creation
    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR, exist_ok=True) 
        print(f"Created directory: {DATA_RAW_DIR}")
        
    
    # 🚨 STEP 2: FETCH PRICE DATA 🚨
    print("\n--- Starting Data Fetchers ---")
    btc_data = fetch_historical_klines(symbol='BTCUSDT', interval='4h', start_str='90 days ago')
    if not btc_data.empty:
        # Saving using the guaranteed absolute path
        btc_data.to_csv(price_file_path)
        print(f"✅ Price Data saved to: {price_file_path}")
    
    
    # 🚨 STEP 3: FETCH NEWS DATA 🚨
    if CRYPTO_NEWS_URL:
        news_data = scrape_crypto_news(CRYPTO_NEWS_URL)
        
        if news_data:
            news_df = pd.DataFrame(news_data)
            # THIS LINE NOW WORKS because news_file_path is defined above!
            news_df.to_csv(news_file_path, index=False) 
            print(f"✅ News Data saved to: {news_file_path}")
        else:
            print(f"⚠️ News scrape failed or returned no data for URL: {CRYPTO_NEWS_URL}")
    else:
        print("Skipping static news scrape due to missing configuration.")
    
    
    print("\n--- Day 2 Data Acquisition Complete ---")


if __name__ == '__main__':
    run_data_fetchers()