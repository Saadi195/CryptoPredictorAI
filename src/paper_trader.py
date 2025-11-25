import sqlite3
import pandas as pd
import os
import sys

# --- 0. PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
# Define the path for the SQLite database file
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'trade_history.db')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

class PaperTradingSystem:
    """Manages the virtual wallet state, trade execution, and logging."""
    def __init__(self, initial_usd=10000.00):
        self.initial_usd = initial_usd
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self._setup_database()
        self.current_usd, self.current_btc = self._load_wallet_state()

    def _setup_database(self):
        """Creates the necessary tables for trade history and wallet state."""
        # Table for transactions (log trade history)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TEXT,
                signal TEXT,
                price REAL,
                btc_change REAL,
                usd_change REAL,
                profit_loss REAL,
                wallet_usd REAL,
                wallet_btc REAL
            )
        """)
        # Table for current wallet state (to easily retrieve balances)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS wallet (
                id INTEGER PRIMARY KEY,
                usd_balance REAL,
                btc_balance REAL
            )
        """)
        self.conn.commit()

    def _load_wallet_state(self):
        """Loads the last saved wallet state or initializes it."""
        self.cursor.execute("SELECT usd_balance, btc_balance FROM wallet WHERE id = 1")
        state = self.cursor.fetchone()
        if state:
            return state[0], state[1]
        else:
            # Initialize wallet for the first time
            self.cursor.execute("INSERT INTO wallet (id, usd_balance, btc_balance) VALUES (?, ?, ?)", 
                                (1, self.initial_usd, 0.0))
            self.conn.commit()
            return self.initial_usd, 0.0
            
    def execute_trade(self, signal, price, trade_amount_usd=500.00):
        """
        Executes a virtual trade (BUY or SELL) using a fixed USD amount 
        and logs the transaction.
        """
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        btc_change = 0.0
        usd_change = 0.0
        profit_loss = 0.0 # Tracked externally for simplicity
        trade_executed = False

        if signal == "BUY" and self.current_usd >= trade_amount_usd:
            # Execute Buy
            btc_bought = trade_amount_usd / price
            self.current_usd -= trade_amount_usd
            self.current_btc += btc_bought
            btc_change = btc_bought
            usd_change = -trade_amount_usd
            trade_executed = True
            print(f"💰 BUY executed: Bought {btc_bought:.6f} BTC @ ${price:,.2f}")
            
        elif signal == "SELL" and self.current_btc * price >= trade_amount_usd:
            # Execute Sell (Liquidate target USD value)
            btc_sold = trade_amount_usd / price
            self.current_usd += trade_amount_usd
            self.current_btc -= btc_sold
            btc_change = -btc_sold
            usd_change = trade_amount_usd
            trade_executed = True
            print(f"🔻 SELL executed: Sold {btc_sold:.6f} BTC @ ${price:,.2f}")
            
        elif signal == "HOLD":
            print("⏳ HOLD signal received. No trade executed.")
            return

        else:
            print(f"⚠️ Trade skipped: Insufficient balance or trade amount too high.")
            return
            
        # Log the trade only if executed
        if trade_executed:
            self.cursor.execute(
                "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (timestamp, signal, price, btc_change, usd_change, profit_loss, self.current_usd, self.current_btc)
            )
            
            # Update wallet state
            self.cursor.execute("UPDATE wallet SET usd_balance = ?, btc_balance = ? WHERE id = 1", 
                                (self.current_usd, self.current_btc))
            self.conn.commit()

    def get_wallet_summary(self, current_price):
        """Calculates current portfolio value and potential profit/loss."""
        portfolio_value = self.current_usd + (self.current_btc * current_price)
        total_profit_loss = portfolio_value - self.initial_usd
        
        print("\n--- Virtual Wallet Summary ---")
        print(f"Initial Capital: ${self.initial_usd:,.2f}")
        print(f"Current USD Balance: ${self.current_usd:,.2f}")
        print(f"Current BTC Balance: {self.current_btc:.6f} BTC")
        print(f"Total Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Total P/L: ${total_profit_loss:,.2f} ({total_profit_loss / self.initial_usd * 100:.2f}%)")
        print("------------------------------")
        return portfolio_value, total_profit_loss

    def get_trade_history(self):
        """Retrieves and displays the full trade history."""
        print("\n--- Trade History ---")
        self.cursor.execute("SELECT timestamp, signal, price, btc_change, usd_change FROM trades")
        history = self.cursor.fetchall()
        
        if not history:
            print("No trades recorded yet.")
            return pd.DataFrame()

        df_history = pd.DataFrame(history, columns=['Timestamp', 'Signal', 'Price', 'BTC Change', 'USD Change'])
        print(df_history.tail())
        return df_history

    def close(self):
        self.conn.close()

# --- Simulation Runner ---

def run_paper_trading_simulation(signal="BUY", current_price=71000.00):
    """Initializes the system and executes a single trade based on input signal."""
    
    # 1. Initialize the system
    trader = PaperTradingSystem()
    
    # 2. Display initial state
    print(f"\nTrading BTC at current price: ${current_price:,.2f}")
    trader.get_wallet_summary(current_price)
    
    # 3. Execute the trade based on the simulated signal
    trader.execute_trade(signal, current_price, trade_amount_usd=500.00) 
    
    # 4. Display final state
    trader.get_wallet_summary(current_price)
    trader.get_trade_history()
    
    trader.close()

if __name__ == '__main__':
    # Execute a sample trade (Run this multiple times to build history for testing)
    run_paper_trading_simulation(signal="BUY", current_price=71000.00)