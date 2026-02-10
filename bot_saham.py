import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
import requests
import os

# Konfigurasi dari GitHub Secrets (Untuk Keamanan)
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def analyze_stocks():
    # Load data emiten dari file Anda
    df_emiten = pd.read_csv('bei_universe.csv')
    candidates = []

    print(f"Memulai screening {len(df_emiten)} saham...")

    for index, row in df_emiten.iterrows():
        symbol = row['symbol']
        try:
            # Ambil data 60 hari terakhir
            data = yf.download(symbol, period="60d", interval="1d", progress=False)
            if len(data) < 30: continue

            # Hitung Indikator menggunakan pandas_ta
            data.ta.rsi(length=14, append=True)
            data.ta.macd(append=True)
            
            # Ambil nilai terakhir
            current_price = data['Close'].iloc[-1]
            rsi = data['RSI_14'].iloc[-1]
            macd_h = data['MACDh_12_26_9'].iloc[-1] # Histogram
            macd_h_prev = data['MACDh_12_26_9'].iloc[-2]
            volume_now = data['Volume'].iloc[-1]
            volume_avg = data['Volume'].tail(5).mean()

            # SISTEM SKORING
            score = 0
            if rsi < 30: score += 3  # Sangat Oversold
            elif rsi < 45: score += 1 # Menuju murah
            
            if macd_h > macd_h_prev: score += 2 # Momentum membaik
            if macd_h > 0 and macd_h_prev < 0: score += 3 # Golden Cross

            if volume_now > (volume_avg * 1.5): score += 2 # Lonjakan Volume

            if score >= 3: # Hanya masukkan jika skor cukup menarik
                candidates.append({
                    'symbol': symbol,
                    'name': row['name'],
                    'price': current_price,
                    'rsi': rsi,
                    'score': score
                })
            
            # Delay agar tidak kena blokir
            time.sleep(0.2) 

        except Exception as e:
            continue

    # Urutkan berdasarkan skor tertinggi, ambil Top 10
    top_10 = sorted(candidates, key=lambda x: x['score'], reverse=True)[:10]

    # Format Pesan
    if top_10:
        msg = "🏆 *TOP 10 WATCHLIST BESOK* 🏆\n\n"
        for i, s in enumerate(top_10, 1):
            msg += f"{i}. *{s['symbol']}* - {s['name']}\n"
            msg += f"   💰 Harga: {s['price']:.0f} | RSI: {s['rsi']:.2f} | Skor: {s['score']}\n\n"
        send_telegram(msg)
    else:
        send_telegram("Market hari ini cenderung sepi, tidak ada saham dengan skor menarik.")

if __name__ == "__main__":
    analyze_stocks()