import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import sys

# --- 0. PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Initialize VADER Sentiment Analyzer
VADER_ANALYZER = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Scores the sentiment of a given text using NLTK's VADER.
    Returns the compound score (-1.0 to +1.0).
    """
    if pd.isna(text) or text is None:
        return 0.0 # Return neutral if text is missing
    
    # Clean text (optional: more sophisticated cleaning can be added here)
    text = str(text).lower().strip()
    
    # Get the VADER scores
    vs = VADER_ANALYZER.polarity_scores(text)
    
    # The 'compound' score is a normalized, weighted composite score.
    return vs['compound']

def aggregate_news_sentiment():
    """
    Loads raw news data, computes sentiment for each headline,
    and returns the average sentiment score.
    """
    news_path = os.path.join(DATA_RAW_DIR, 'crypto_news_static.csv')
    print(f"\n📰 Starting NLP Sentiment Analysis on: {news_path}")
    
    if not os.path.exists(news_path):
        print("❌ Error: Raw news data not found. Returning neutral sentiment (0.0).")
        return 0.0

    df_news = pd.read_csv(news_path)
    
    if df_news.empty or 'headline' not in df_news.columns:
        print("⚠️ News data is empty or missing 'headline' column. Returning neutral sentiment (0.0).")
        return 0.0
        
    # Apply the sentiment analyzer to each headline
    df_news['sentiment_score'] = df_news['headline'].apply(analyze_sentiment)
    
    # Aggregate: Calculate the mean of all sentiment scores
    # This gives us a single number representing the overall market sentiment from the latest news batch.
    avg_sentiment = df_news['sentiment_score'].mean()
    
    print(f"✅ NLP Analysis Complete. Overall Sentiment Score: {avg_sentiment:.4f}")
    return avg_sentiment

def integrate_sentiment_into_features(avg_sentiment):
    """
    Loads the main feature file and updates the News_Sentiment column.
    """
    features_path = os.path.join(DATA_PROCESSED_DIR, 'btc_ml_features.csv')
    print(f"\nMerging sentiment score into {features_path}")

    if not os.path.exists(features_path):
        print(f"❌ Error: ML features file not found at {features_path}. Run Day 4 first!")
        return
        
    df_features = pd.read_csv(features_path, index_col='Open Time', parse_dates=True)
    
    # Overwrite the placeholder value (0.5) with the real sentiment score
    df_features['News_Sentiment'] = avg_sentiment
    
    # Save the final, complete feature set
    df_features.to_csv(features_path)
    
    print("✅ Sentiment integration successful.")
    print(f"Sample row (Last 3 columns):")
    print(df_features[['FNG_Index', 'News_Sentiment', 'target']].tail(1))
    
    
def run_sentiment_pipeline():
    """Main function to run the entire Day 5 pipeline."""
    # Step 1: Compute Sentiment Score
    sentiment_score = aggregate_news_sentiment()
    
    # Step 2: Integrate Score into the ML Dataset
    integrate_sentiment_into_features(sentiment_score)


if __name__ == '__main__':
    run_sentiment_pipeline()