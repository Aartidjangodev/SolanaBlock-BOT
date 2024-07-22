import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import os
from base58 import b58decode
from dotenv import load_dotenv
from solanasdk.rpc.api import Client
from solanasdk.rpc.types import TxOpts
from solanasdk.transaction import Transaction
from solanasdk.system_program import transfer, TransferParams
from solanasdk.keypair import Keypair
from solanasdk.publickey import PublicKey
import logging

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='trading_bot.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Define API endpoint, payload, and headers
url = "https://streaming.bitquery.io/eap"
payload = json.dumps({
    "query": """query MyQuery {
  Solana(network: solana, dataset: realtime) {
    DEXTrades {
      ChainId
      Trade {
        Buy {
          AmountInUSD
          Amount
          Currency {
            Name
            Symbol
            MintAddress
          }
          Price
          PriceInUSD
          Account {
            Address
          }
        }
        Dex {
          ProtocolName
        }
        Market {
          MarketAddress
        }
        Sell {
          Amount
          AmountInUSD
          Currency {
            Name
            Symbol
            MintAddress
          }
          Price
          PriceInUSD
          Account {
            Address
          }
        }
      }
      Transaction {
        Fee
      }
    }
  }
}""",
    "variables": "{}"
})

headers = {
    'Content-Type': 'application/json',
    'X-API-KEY': 'BQYV6Z4MPu7JH4okaYH8yhl6V7quCgXS',
    'Authorization': 'Bearer ory_at_EPoO8Kiz3pHWxGnhdDoGBOGz9x5IiinjPlePi3CjgAU.8m2uF9mgfNYq2OOKq_v1E3MnmvvFWs99YWkHJ29C3rM'
}

# Solana configuration
SOLANA_RPC_ENDPOINT = "https://api.testnet.solana.com"
PHANTOM_WALLET_PRIVATE_KEY = os.getenv("wallet_private_key")
TRADE_AMOUNT = int(0.03 * 10**9) 


# Initialize dictionaries to store timestamps of transactions and volume
txn_times = {
    'm5': {'buys': [], 'sells': []},
    'h1': {'buys': [], 'sells': []},
    'h6': {'buys': [], 'sells': []},
    'h24': {'buys': [], 'sells': []}
}

volume_times = {
    'm5': {'buys': 0, 'sells': 0},
    'h1': {'buys': 0, 'sells': 0},
    'h6': {'buys': 0, 'sells': 0},
    'h24': {'buys': 0, 'sells': 0}
}

# Initialize the txns and volumes dictionaries to store transaction counts and volumes
txns = {
    'm5': {'buys': 0, 'sells': 0},
    'h1': {'buys': 0, 'sells': 0},
    'h6': {'buys': 0, 'sells': 0},
    'h24': {'buys': 0, 'sells': 0}
}

volumes = {
    'm5': {'buys': 0, 'sells': 0},
    'h1': {'buys': 0, 'sells': 0},
    'h6': {'buys': 0, 'sells': 0},
    'h24': {'buys': 0, 'sells': 0}
}

# Sample data for demonstration
volume_data = {
    'timestamp': [],
    'close': [],
    'open': [],
    'volume': []
}




# Inputs for MA and Std calculation
length = 610  
slength = 610 
thresholdExtraHigh = 3  
thresholdHigh = 2.5 
thresholdMedium = 0.5  
thresholdNormal = -0.5  

# Counter for non-Wrapped Solana trades
non_wrapped_solana_count = 0
total_trades = 0
total_wins = 0
total_losses = 0

def highlight_value(value, context):
    """Return the value highlighted with appropriate color based on its value."""
    try:
        value = float(value)
    except ValueError:
        return f"{value} ({context})"

    if value <= 1000:
        # Red background for value <= 1000
        return f"\033[41m\033[97m{value} ({context})\033[0m"
    else:
        # Blue background for value > 1000
        return f"\033[44m\033[97m{value} ({context})\033[0m"


def update_counters(trade_type, amount):
    """Update transaction and volume counters for different timeframes."""
    current_time = datetime.now()
    timeframes = {
        'm5': timedelta(minutes=5),
        'h1': timedelta(hours=1),
        'h6': timedelta(hours=6),
        'h24': timedelta(hours=24)
    }

    for timeframe, delta in timeframes.items():
        # Remove outdated transactions
        txn_times[timeframe]['buys'] = [
            t for t in txn_times[timeframe]['buys'] if t > current_time - delta]
        txn_times[timeframe]['sells'] = [
            t for t in txn_times[timeframe]['sells'] if t > current_time - delta]

        # Update volumes
        if trade_type == "Buy":
            volumes[timeframe]['buys'] = sum(
                amount for t in txn_times[timeframe]['buys'])
            txn_times[timeframe]['buys'].append(current_time)
        else:
            volumes[timeframe]['sells'] = sum(
                amount for t in txn_times[timeframe]['sells'])
            txn_times[timeframe]['sells'].append(current_time)

        # Update counters
        txns[timeframe]['buys'] = len(txn_times[timeframe]['buys'])
        txns[timeframe]['sells'] = len(txn_times[timeframe]['sells'])


def update_indicators(df):
    """Update indicators like moving average, standard deviation, and conditions."""
    length_used = min(length, len(df))
    slength_used = min(slength, len(df))

    if len(df) > 0:
        # Calculate moving average and standard deviation
        df['mean'] = df['volume'].rolling(
            window=length_used, min_periods=1).mean()
        df['std'] = df['volume'].rolling(
            window=slength_used, min_periods=1).std()

        # Calculate stdbar
        df['stdbar'] = (df['volume'] - df['mean']) / df['std']

        # Determine volume thresholds
        df['bcolor'] = np.where(df['stdbar'] > thresholdExtraHigh,
                                np.where(
                                    df['close'] > df['open'], 'cthresholdExtraHighUp', 'cthresholdExtraHighDn'),
                                np.where(df['stdbar'] > thresholdHigh,
                                         np.where(
                                             df['close'] > df['open'], 'cthresholdHighUp', 'cthresholdHighDn'),
                                         np.where(df['stdbar'] > thresholdMedium,
                                                  np.where(
                                                      df['close'] > df['open'], 'cthresholdMediumUp', 'cthresholdMediumDn'),
                                                  np.where(df['stdbar'] > thresholdNormal,
                                                           np.where(
                                                               df['close'] > df['open'], 'cthresholdNormalUp', 'cthresholdNormalDn'),
                                                           np.where(df['close'] > df['open'], 'cthresholdLowUp', 'cthresholdLowDn')))))

        # Define conditions for alerts
        df['conditionExtraHigh'] = df['stdbar'] > thresholdExtraHigh
        df['conditionHigh'] = df['stdbar'] > thresholdHigh
        df['conditionMedium'] = df['stdbar'] > thresholdMedium
        df['conditionNormal'] = df['stdbar'] > thresholdNormal

        # Define buy and sell signals
        df['Buy_Signal'] = df['conditionExtraHigh'] | df['conditionHigh'] | df['conditionMedium']
        df['Sell_Signal'] = ~df['Buy_Signal'] & df['conditionNormal']
        # Calculate heatmap volume
        df['heatmap_volume'] = df['volume'] * df['stdbar']
    return df


def adaptive_moving_average(src, length=14, fastLength=2, slowLength=30):
    fastAlpha = 2 / (fastLength + 1)
    slowAlpha = 2 / (slowLength + 1)

    def highest(data, period):
        return np.max(data[-period:])

    def lowest(data, period):
        return np.min(data[-period:])

    ama = np.zeros_like(src)
    for i in range(1, len(src)):
        hh = highest(src[:i+1], length + 1)
        ll = lowest(src[:i+1], length + 1)
        mltp = abs(2 * src[i] - ll - hh) / (hh - ll) if (hh - ll) != 0 else 0
        ssc = mltp * (fastAlpha - slowAlpha) + slowAlpha
        ama[i] = ama[i-1] + (ssc ** 2) * (src[i] - ama[i-1])
    return ama


def highlight_value(value, context):
    """Return the value highlighted with appropriate color based on its value."""
    try:
        value = float(value)
    except ValueError:
        return f"{value} ({context})"

    if value <= 1000:
        # Red background for value <= 1000
        return f"\033[41m\033[97m{value} ({context})\033[0m"
    else:
        # Blue background for value > 1000
        return f"\033[44m\033[97m{value} ({context})\033[0m"


# Initialize a dictionary to track market addresses and their price histories
market_price_history = {}
token_addresses = {}


def is_honeypot(market_address, buy_token_address, sell_token_address, price, buy_amount, sell_amount):
    """Check if the given market address or token addresses exhibit honeypot characteristics."""
    # Initialize the market in the price history dictionary if not already present
    if market_address not in market_price_history:
        market_price_history[market_address] = []

    # Initialize token addresses if not already present
    if buy_token_address not in token_addresses:
        token_addresses[buy_token_address] = {"buy": 0, "sell": 0}
    if sell_token_address not in token_addresses:
        token_addresses[sell_token_address] = {"buy": 0, "sell": 0}

    # Append the current price to the market's price history
    market_price_history[market_address].append(price)

    # Update token addresses trading volumes
    token_addresses[buy_token_address]["buy"] += buy_amount
    token_addresses[sell_token_address]["sell"] += sell_amount

    # Check for price fluctuation
    if len(market_price_history[market_address]) > 1:
        price_changes = [abs(market_price_history[market_address][i] - market_price_history[market_address][i - 1])
                         for i in range(1, len(market_price_history[market_address]))]
        average_price_change = sum(price_changes) / len(price_changes)
        if average_price_change > 0.1 * price:  # Example threshold for price fluctuation
            return True

    # Check for low liquidity or suspicious trading volumes
    # Example threshold for suspicious buy volume
    if token_addresses[buy_token_address]["buy"] > 10 * token_addresses[buy_token_address]["sell"]:
        return True
    # Example threshold for suspicious sell volume
    if token_addresses[sell_token_address]["sell"] > 10 * token_addresses[sell_token_address]["buy"]:
        return True

    return False


neon_colors = {
    "Neon Green": "\033[38;2;57;255;20m",
    "Neon Blue": "\033[38;2;13;0;255m",
    "Neon Pink": "\033[38;2;255;110;199m",
    "Neon Yellow": "\033[38;2;255;255;51m",
    "Neon Orange": "\033[38;2;255;103;0m",
    "Neon Red": "\033[38;2;255;7;58m",
    "Neon Purple": "\033[38;2;176;38;255m",
    "Neon Cyan": "\033[38;2;0;255;255m",
    "Neon Lime": "\033[38;2;175;255;0m",
    "Neon Magenta": "\033[38;2;255;0;255m",
    "Neon White": "\033[38;2;255;255;255m",
    "Neon Teal": "\033[38;2;0;255;187m",
    "Neon Violet": "\033[38;2;157;0;255m",
    "Neon Turquoise": "\033[38;2;0;255;239m",
    "Neon Coral": "\033[38;2;255;67;101m",
    "Neon Chartreuse": "\033[38;2;223;255;0m"
}


def neon_print(message, color):
    reset = '\033[0m'
    print(f'{color}{message}{reset}')

def adaptive_moving_average(src, length=14, fastLength=2, slowLength=30):
    fastAlpha = 2 / (fastLength + 1)
    slowAlpha = 2 / (slowLength + 1)

    def highest(data, period):
        return np.max(data[-period:])

    def lowest(data, period):
        return np.min(data[-period:])

    ama = np.zeros_like(src)
    for i in range(1, len(src)):
        hh = highest(src[:i+1], length + 1)
        ll = lowest(src[:i+1], length + 1)
        mltp = abs(2 * src[i] - ll - hh) / (hh - ll) if (hh - ll) != 0 else 0
        ssc = mltp * (fastAlpha - slowAlpha) + slowAlpha
        ama[i] = ama[i-1] + (ssc ** 2) * (src[i] - ama[i-1])
    return ama

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr

def place_trade(signal_type, current_price, stop_loss_ratio, take_profit_ratio, slippage=0.50):
    client = Client(SOLANA_RPC_ENDPOINT)
    
    if PHANTOM_WALLET_PRIVATE_KEY is None:
        log_error("Error: PHANTOM_WALLET_PRIVATE_KEY environment variable not set.")
        return None
    
    try:
        private_key_bytes = b58decode(PHANTOM_WALLET_PRIVATE_KEY)
        sender_keypair = Keypair.from_secret_key(private_key_bytes)
    except Exception as e:
        log_error(f"Error decoding private key: {e}")
        return None
    
    receiver_pubkey = sender_keypair.public_key
    
    # Adjust trade amount for slippage
    lamports = int(TRADE_AMOUNT * (1 - slippage))

    transaction = Transaction().add(
        transfer(
            TransferParams(
                from_pubkey=sender_keypair.public_key,
                to_pubkey=receiver_pubkey,
                lamports=lamports
            )
        )
    )

    try:
        response = client.send_transaction(transaction, sender_keypair, opts=TxOpts(skip_preflight=True))

        # Calculate stop loss and take profit levels
        stop_loss = current_price - (current_price * stop_loss_ratio)
        take_profit = current_price + (current_price * take_profit_ratio)

        return response, current_price, stop_loss, take_profit
    except Exception as e:
        log_error(f"Error placing trade: {e}")
        return None, None, None, None

def analyze_market(df):
    print("*" * 30)
    log_info("Analyzing the market...")

    # Simulate market analysis process
    time.sleep(10)  # Placeholder for actual market analysis code

    # Calculate Adaptive Moving Average (AMA)
    df['AMA'] = adaptive_moving_average(df['close'].values)
    
    # Calculate ATR (Average True Range)
    df['ATR'] = calculate_atr(df)

    current_price = df['close'].iloc[-1]
    ama_value = df['AMA'].iloc[-1]

    # Determine the trade signal based on AMA
    if ama_value > current_price:
        return 'buy'
    elif ama_value < current_price:
        return 'sell'
    else:
        return 'hold'

def update_performance_metrics(trade_type, profit):
    global total_trades, total_wins, total_losses
    total_trades += 1
    if profit > 0:
        total_wins += 1
    else:
        total_losses += 1
    logging.info(f"Total Trades: {total_trades}, Wins: {total_wins}, Losses: {total_losses}")


def log_error(message):
    logging.error(message)

def log_info(message):
    logging.info(message)
    
def log_warning(message):
    logging.warning(message)

def format_trade_response(response, token_name, signal_type):
    response_details = {
        "Token": token_name,
        "Trade Type": signal_type.capitalize(),
        "Response": response
    }
    return json.dumps(response_details, indent=4)    

def fetch_and_print_trades():
    global non_wrapped_solana_count, total_trades, total_wins, total_losses
    response = requests.post(url, headers=headers, data=payload)

    # Check the response status code
    if response.status_code == 200:
        data = response.json()

        if response.text:
            try:
                data = response.json()
            except json.JSONDecodeError:
                log_error("Error: Response content is not valid JSON")
                log_warning("Response content:", response.text)
                return

            if 'data' in data and 'Solana' in data['data'] and 'DEXTrades' in data['data']['Solana']:
                dex_trades = data['data']['Solana']['DEXTrades']

                total_market_cap = 0
                for trade in dex_trades:
                    chain_id = trade['ChainId']
                    dex = trade['Trade']['Dex']
                    buy = trade['Trade']['Buy']
                    sell = trade['Trade']['Sell']

                    buy_currency_name = buy['Currency']['Name']
                    buy_mint_address = buy['Currency']['MintAddress']
                    sell_currency_name = sell['Currency']['Name']
                    sell_mint_address = sell['Currency']['MintAddress']
                    buy_amount = float(buy['Amount'])
                    buy_price = float(buy['Price'])
                    sell_amount = float(sell['Amount'])
                    sell_price = float(sell['Price'])
                    trade_type = "Buy" if buy_currency_name != "Wrapped Solana" else "Sell"

                    if non_wrapped_solana_count < 10 and (buy_currency_name == "Wrapped Solana" or sell_currency_name == "Wrapped Solana"):
                        continue

                    if non_wrapped_solana_count < 10:
                        non_wrapped_solana_count += 1

                    profit = buy_amount - sell_amount
                    loss = sell_amount - buy_amount

                    candle_color = "green" if buy_price > sell_price else "red"

                    update_counters(trade_type, buy_amount if trade_type == "Buy" else sell_amount)

                    volume_data['timestamp'].append(datetime.now())
                    volume_data['close'].append(buy_price if trade_type == "Buy" else sell_price)
                    volume_data['open'].append(buy_price if trade_type == "Buy" else sell_price)
                    volume_data['volume'].append(buy_amount if trade_type == "Buy" else sell_amount)

                    buy_market_cap = buy_amount * buy_price
                    sell_market_cap = sell_amount * sell_price
                    net_market_cap_change = buy_market_cap - sell_market_cap

                    total_market_cap += net_market_cap_change

                    liquidity = buy_amount + sell_amount
                    liquidity_status = "Locked 🔒" if liquidity > 0 else "Unlocked 📫"

                    rug_check = '✅' if liquidity > 100000 or 'pump' in dex['ProtocolName'].lower() else '❌'

                    df = pd.DataFrame(volume_data)

                    df['high'] = df[['open', 'close']].max(axis=1)
                    df['low'] = df[['open', 'close']].min(axis=1)

                    df['AMA'] = adaptive_moving_average(df['close'].values)

                    df['ATR'] = calculate_atr(df)

                    trade_signal = analyze_market(df)
                    current_price = df['close'].iloc[-1]
                    ama_value = df['AMA'].iloc[-1]
                    atr_value = df['ATR'].iloc[-1]
                   

                    print("-" * 30)

                    if trade_signal == 'buy':
                        stop_loss = df['low'].iloc[-4:].min()
                    elif trade_signal == 'sell':
                        stop_loss = df['high'].iloc[-4:].max()

                    take_profit_ratio = 0.25
                    take_profit = current_price + (current_price * take_profit_ratio)

                    print("-" * 30)

                    if trade_signal == 'buy':
                        response, trade_price, sl, tp = place_trade('buy', current_price, stop_loss, take_profit)
                        if response:
                            formatted_response = format_trade_response(response, buy_currency_name, 'buy')
                            log_info(f"💼 Trade Type: Buy\n🔢 ATR: {atr_value}\n💵 Profit/Loss: 0.000000 SOL\n💰Current Price: {current_price} SOL\n💲Trade Placed: Buy trade placed successfully\n💰Buy Price: {trade_price}\nStop Loss: {sl}\nTake Profit: {tp}\n📈Price Movement: {'green' if buy_price > sell_price else 'red'}\n🔔 AMA Signal: {ama_value}\n")
                            log_info(f"✅Transaction: {formatted_response}")
                            update_performance_metrics('buy', profit)
                        else:
                            log_warning("Buy trade failed.")
                    elif trade_signal == 'sell':
                        response, trade_price, sl, tp = place_trade('sell', current_price, stop_loss, take_profit)
                        if response:
                            formatted_response = format_trade_response(response, sell_currency_name, 'sell')
                            log_info(f"💼 Trade Type: Sell\n🔢 ATR: {atr_value}\n💵 Profit/Loss: 0.000000 SOL\n💰Current Price: {current_price} SOL\n💲Trade Placed: Sell trade placed successfully\n💰Sell Price: {trade_price}\nStop Loss: {sl}\nTake Profit: {tp}\n📈Price Movement: {'green' if buy_price > sell_price else 'red'}\n🔔 AMA Signal: {ama_value}\n")
                            log_info(f"✅Transaction: {formatted_response}")
                            update_performance_metrics('sell',profit)
                        else:
                            log_warning("Sell trade failed.")
                    else:
                        log_info(f"🔔 AMA Signal: {ama_value}\n📈Price Movement: {'green' if buy_price > sell_price else 'red'}\n⏳Hold Signal Detected.")
                    print("-" * 30) 
                    time.sleep(30)       
            else:
                log_warning("No data available in the response")
        else:
            log_warning("Error: Received empty response")
    else:
        log_warning(f"Error: Received response with status code {response.status_code}")
        print("Response content:", response.text)
    time.sleep(10)
    
log_info("Starting trading bot...")
# Run the fetch and print function in a loop
while True:
    fetch_and_print_trades()
