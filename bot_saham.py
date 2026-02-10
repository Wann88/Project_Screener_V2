import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time
import requests
import os

# Ambil data dari GitHub Secrets
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except:
        print("Gagal mengirim pesan ke Telegram")

def analyze_stocks():
    # Load 956 emiten dari file Anda
    try:
        df_emiten = pd.read_csv('bei_universe.csv')
    except:
        print("File bei_universe.csv tidak ditemukan!")
        return

    candidates = []
    print(f"Mulai scanning {len(df_emiten)} saham...")

    for index, row in df_emiten.iterrows():
        symbol = row['symbol']
        try:
            # Ambil data 60 hari untuk perhitungan MA dan RSI
            data = yf.download(symbol, period="60d", interval="1d", progress=False)
            if len(data) < 30: continue

            # Hitung Indikator
            data.ta.rsi(length=14, append=True)
            data.ta.macd(append=True)
            
            # Nilai hari ini dan sebelumnya
            price_now = data['Close'].iloc[-1]
            rsi_now = data['RSI_14'].iloc[-1]
            macd_h = data['MACDh_12_26_9'].iloc[-1]
            macd_h_prev = data['MACDh_12_26_9'].iloc[-2]
            vol_now = data['Volume'].iloc[-1]
            vol_avg = data['Volume'].tail(5).mean()

            # LOGIKA SKORING (Sederhana tapi Powerfull)
            score = 0
            # 1. Kondisi RSI
            if rsi_now < 30: score += 4  # Sangat Oversold
            elif rsi_now < 40: score += 2 # Menuju Murah
            
            # 2. Kondisi MACD (Momentum)
            if macd_h > macd_h_prev: score += 2 # Momentum menguat
            if macd_h > 0 and macd_h_prev < 0: score += 3 # Golden Cross

            # 3. Kondisi Volume
            if vol_now > (vol_avg * 1.5): score += 2 # Volume di atas rata-rata

            if score >= 4: # Hanya simpan yang punya sinyal cukup kuat
                candidates.append({
                    'symbol': symbol,
                    'name': row['name'],
                    'price': price_now,
                    'rsi': rsi_now,
                    'score': score
                })
            
            # Delay kecil agar tidak diblokir Yahoo Finance
            time.sleep(0.3) 

        except:
            continue

    # Urutkan berdasarkan Skor Tertinggi, ambil Top 10
    top_10 = sorted(candidates, key=lambda x: x['score'], reverse=True)[:10]

    # Format Pesan Telegram
    if top_10:
        msg = "🏆 *TOP 10 WATCHLIST BESOK* 🏆\n\n"
        for i, s in enumerate(top_10, 1):
            msg += f"{i}. *{s['symbol']}* - {s['name']}\n"
            msg += f"   💰 Harga: {s['price']:.0f} | RSI: {s['rsi']:.2f} | Skor: {s['score']}\n\n"
        send_telegram(msg)
    else:
        send_telegram("Tidak ditemukan saham dengan sinyal menarik hari ini.")

if __name__ == "__main__":
    analyze_stocks()