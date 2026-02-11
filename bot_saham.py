import pandas as pd
import yfinance as yf
import time
import requests
import os
import sys

# Coba import pandas_ta, jika gagal pakai fallback manual
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
    print("Menggunakan library: pandas-ta")
except ImportError:
    HAS_PANDAS_TA = False
    print("pandas-ta tidak ditemukan. Menggunakan perhitungan manual (Fallback Mode).")

# Konfigurasi
BATCH_SIZE = 50  # Jumlah saham per batch unduhan
# Ambil data dari GitHub Secrets
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram(message):
    """Mengirim pesan ke Telegram dengan penanganan error."""
    if not TOKEN or not CHAT_ID:
        # print("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID belum diset.")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Gagal mengirim pesan ke Telegram: {e}")

def calculate_technical(df):
    """Menghitung indikator teknikal (Support pandas-ta & Manual)."""
    if df.empty or len(df) < 50:
        return None
    
    # Copy untuk menghindari peringatan SettingWithCopy
    df = df.copy()

    if HAS_PANDAS_TA:
        # --- MENGGUNAKAN PANDAS-TA (JIKA ADA) ---
        # 1. RSI 14
        df['RSI'] = df.ta.rsi(length=14)
        
        # 2. MACD (12, 26, 9)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
        
        # 3. Moving Averages
        df['SMA_200'] = df.ta.sma(length=200)
        df['SMA_50'] = df.ta.sma(length=50)

    else:
        # --- MANUAL CALCULATION (FALLBACK) ---
        # 1. RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. MACD (12, 26, 9)
        k = df['Close'].ewm(span=12, adjust=False).mean()
        d = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26'] = k - d
        # MACDh_12_26_9 adalah Histogram (MACD - Signal)
        # Note: Nama kolom harus disamakan manual agar logic di bawah tetap jalan
        df['MACDh_12_26_9'] = df['MACD_12_26'] - df['MACD_12_26'].ewm(span=9, adjust=False).mean() 
        
        # 3. Moving Averages
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # 4. Volume Moving Average (Sama untuk kedua mode)
    df['VOL_MA5'] = df['Volume'].rolling(window=5).mean()

    return df

def process_batch(tickers):
    """Memproses satu batch ticker."""
    candidates = []
    
    print(f"Mengunduh data untuk {len(tickers)} saham...")
    
    try:
        # Download data bulk
        data = yf.download(tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        print(f"Error download batch: {e}")
        return []

    # Handle single ticker result format
    if len(tickers) == 1:
        data = {tickers[0]: data}

    # Iterasi setiap ticker dalam batch
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data[ticker]
            else:
                try:
                    df = data[ticker].dropna(how='all')
                except KeyError:
                    continue
            
            if df.empty or len(df) < 60:
                continue

            # Hitung Indikator
            df = calculate_technical(df)
            if df is None:
                continue

            # Ambil data hari terakhir
            curr = df.iloc[-1]
            prev = df.iloc[-2]

            # --- LOGIKA SKORING ---
            score = 0
            reasons = []

            # 1. Filter Likuiditas Minimum (Harga > 50 & Volume > 1000 lembar)
            if curr['Close'] < 50 or curr['Volume'] < 1000:
                continue

            # 2. RSI Oversold tapi Mulai Rebound
            # Pastikan kolom RSI ada (pandas-ta kadang return NaN di awal data, tapi kita ambil iloc[-1])
            if pd.isna(curr['RSI']): continue
            
            if curr['RSI'] < 30:
                score += 3
                reasons.append("RSI Oversold (<30)")
            elif 30 <= curr['RSI'] < 40:
                score += 1
                reasons.append("RSI Murah (<40)")
            
            # 3. MACD Golden Cross atau Menguat
            # Pastikan kolom MACD ada
            if 'MACDh_12_26_9' in df.columns:
                macd_h_now = curr['MACDh_12_26_9']
                macd_h_prev = prev['MACDh_12_26_9']
                
                if macd_h_now > 0 and macd_h_prev < 0:
                    score += 4
                    reasons.append("MACD Golden Cross")
                elif macd_h_now > macd_h_prev and macd_h_now > -0.5: 
                    score += 1

            # 4. Volume Spike (Indikasi Akumulasi)
            if curr['Volume'] > (curr['VOL_MA5'] * 1.5):
                score += 2
                reasons.append("Volume Spike (>1.5x Avg)")

            # 5. Trend Filter (Opsional: Bonus poin jika di atas MA200)
            if pd.notna(curr['SMA_200']) and curr['Close'] > curr['SMA_200']:
                score += 1
                reasons.append("Uptrend (Above MA200)")

            # Kriteria Lolos: Skor Minimal 4
            if score >= 4:
                candidates.append({
                    'symbol': ticker,
                    'price': curr['Close'],
                    'rsi': curr['RSI'],
                    'volume': curr['Volume'],
                    'score': score,
                    'reasons': ", ".join(reasons)
                })

        except Exception as e:
            # print(f"Error {ticker}: {e}")
            continue
            
    return candidates

def main():
    start_time = time.time()
    
    # 1. Load Data Saham
    try:
        df_emiten = pd.read_csv('bei_universe.csv')
        all_tickers = df_emiten['symbol'].tolist()
        all_tickers = [x if x.endswith('.JK') else f"{x}.JK" for x in all_tickers]
    except FileNotFoundError:
        print("File bei_universe.csv tidak ditemukan!")
        return

    print(f"Total saham deteksi: {len(all_tickers)}")
    
    unique_tickers = list(set(all_tickers)) # Hapus duplikat
    
    final_candidates = []
    
    # 2. Proses per Batch
    for i in range(0, len(unique_tickers), BATCH_SIZE):
        batch = unique_tickers[i:i+BATCH_SIZE]
        print(f"Memproses batch {i//BATCH_SIZE + 1}/{(len(unique_tickers)//BATCH_SIZE)+1}...")
        
        candidates = process_batch(batch)
        final_candidates.extend(candidates)
        
        # Jeda singkat antar batch
        time.sleep(1)

    # 3. Urutkan & Filter Top Picks
    final_candidates.sort(key=lambda x: x['score'], reverse=True)
    top_picks = final_candidates[:15]  # Ambil Top 15

    # 4. Kirim Laporan
    if top_picks:
        msg = "🚀 *SCREENER SAHAM PREMIUM* 🚀\n"
        mode_text = "pandas-ta" if HAS_PANDAS_TA else "Manual-Mode"
        msg += f"_Scanned {len(unique_tickers)} stocks ({mode_text}) in {int(time.time() - start_time)}s_\n\n"
        
        for i, stock in enumerate(top_picks, 1):
            icon = "🔥" if stock['score'] >= 6 else "✅"
            msg += f"{icon} *{stock['symbol']}* (Skor: {stock['score']})\n"
            msg += f"   M: {stock['price']:.0f} | RSI: {stock['rsi']:.1f}\n"
            msg += f"   Sinyal: _{stock['reasons']}_\n\n"
            
        if len(msg) > 4000:
            msg = msg[:4000] + "\n...(terpotong)"
            
        send_telegram(msg)
        print("Laporan terkirim ke Telegram.")
    else:
        send_telegram("Mencoba screening... Tidak ada saham yang memenuhi kriteria ketat hari ini.")
        print("Tidak ada kandidat hari ini.")

if __name__ == "__main__":
    main()
