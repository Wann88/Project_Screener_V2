import pandas as pd
import yfinance as yf
import time
import requests
import os
import sys
import numpy as np
import logging
from datetime import datetime

# ============================================================
# SETUP LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log', encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)

# Coba import pandas_ta, jika gagal pakai fallback manual
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
    log.info("Menggunakan library: pandas-ta")
except ImportError:
    HAS_PANDAS_TA = False
    log.info("pandas-ta tidak ditemukan. Menggunakan perhitungan manual (Fallback Mode).")

# ============================================================
# KONFIGURASI
# ============================================================
BATCH_SIZE = 50
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Threshold skor berdasarkan market regime
SCORE_THRESHOLDS = {
    'BULLISH': 5,
    'NEUTRAL': 6,
    'BEARISH': 8,
}

# Likuiditas minimum
MIN_PRICE = 100        # Minimum harga saham (Rp)
MIN_VOLUME_AVG = 100_000  # Minimum rata-rata volume 20 hari

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message):
    """Mengirim pesan ke Telegram. Otomatis split jika terlalu panjang."""
    if not TOKEN or not CHAT_ID:
        log.warning("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID belum diset. Pesan tidak dikirim.")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    
    # Split pesan jika lebih dari 4000 karakter
    messages = _split_message(message, max_len=4000)
    
    for i, msg in enumerate(messages):
        if len(messages) > 1:
            header = f"📄 _Part {i+1}/{len(messages)}_\n\n"
            msg = header + msg
        
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            log.info(f"Pesan Telegram terkirim (part {i+1}/{len(messages)})")
        except Exception as e:
            log.error(f"Gagal mengirim pesan ke Telegram (part {i+1}): {e}")
        
        if i < len(messages) - 1:
            time.sleep(0.5)  # Rate limit protection


def _split_message(text, max_len=4000):
    """Split pesan menjadi beberapa bagian berdasarkan double newline."""
    if len(text) <= max_len:
        return [text]
    
    parts = []
    current = ""
    
    # Split by stock entry (double newline)
    blocks = text.split("\n\n")
    
    for block in blocks:
        if len(current) + len(block) + 2 > max_len:
            if current:
                parts.append(current.strip())
            current = block
        else:
            current = current + "\n\n" + block if current else block
    
    if current.strip():
        parts.append(current.strip())
    
    return parts if parts else [text[:max_len]]


def send_error_alert(error_msg):
    """Kirim alert ke Telegram jika bot error/crash."""
    if not TOKEN or not CHAT_ID:
        return
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = (
        f"🚨 *BOT ERROR ALERT* 🚨\n\n"
        f"⏰ Waktu: `{timestamp}`\n"
        f"❌ Error: `{error_msg[:500]}`\n\n"
        f"_Bot gagal menyelesaikan screening._"
    )
    
    try:
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=15)
    except Exception:
        pass  # Last resort, nothing we can do


# ============================================================
# MARKET REGIME DETECTION
# ============================================================
def get_market_regime():
    """
    Deteksi kondisi market (IHSG) untuk menyesuaikan threshold.
    Returns: ('BULLISH'|'NEUTRAL'|'BEARISH', detail_string)
    """
    try:
        log.info("Mengunduh data IHSG (^JKSE)...")
        ihsg = yf.download('^JKSE', period='1y', interval='1d', auto_adjust=True, progress=False)
        
        if ihsg.empty or len(ihsg) < 200:
            log.warning("Data IHSG tidak cukup. Default: NEUTRAL")
            return 'NEUTRAL', 'Data IHSG tidak tersedia'
        
        # Flatten MultiIndex columns if present
        if isinstance(ihsg.columns, pd.MultiIndex):
            ihsg.columns = ihsg.columns.get_level_values(0)
        
        close = ihsg['Close'].iloc[-1]
        sma50 = ihsg['Close'].rolling(50).mean().iloc[-1]
        sma200 = ihsg['Close'].rolling(200).mean().iloc[-1]
        
        # RSI IHSG
        delta = ihsg['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        ihsg_rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        change_5d = ((close - ihsg['Close'].iloc[-6]) / ihsg['Close'].iloc[-6] * 100) if len(ihsg) >= 6 else 0
        
        # Determine regime
        if close > sma50 and close > sma200 and sma50 > sma200:
            regime = 'BULLISH'
        elif close < sma50 and close < sma200:
            regime = 'BEARISH'
        else:
            regime = 'NEUTRAL'
        
        detail = (
            f"IHSG: {close:,.0f} | SMA50: {sma50:,.0f} | SMA200: {sma200:,.0f}\n"
            f"RSI: {ihsg_rsi:.1f} | 5D Change: {change_5d:+.2f}%"
        )
        
        log.info(f"Market Regime: {regime} ({detail})")
        return regime, detail
        
    except Exception as e:
        log.error(f"Error deteksi market regime: {e}")
        return 'NEUTRAL', f'Error: {e}'


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def calculate_technical(df):
    """Menghitung semua indikator teknikal (manual, tanpa dependency tambahan)."""
    if df.empty or len(df) < 50:
        return None
    
    df = df.copy()
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ---- 1. RSI 14 ----
    if HAS_PANDAS_TA:
        df['RSI'] = df.ta.rsi(length=14)
    else:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # ---- 2. MACD (12, 26, 9) ----
    if HAS_PANDAS_TA:
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None:
            df = pd.concat([df, macd], axis=1)
    else:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26'] = ema12 - ema26
        df['MACDh_12_26_9'] = df['MACD_12_26'] - df['MACD_12_26'].ewm(span=9, adjust=False).mean()

    # ---- 3. Moving Averages ----
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # ---- 4. ATR (Average True Range) ----
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # ---- 5. Volume Indicators ----
    df['VOL_MA5'] = df['Volume'].rolling(window=5).mean()
    df['VOL_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # OBV (On Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()

    # ---- 6. Bollinger Bands (20, 2) ----
    df['BB_MID'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MID'] + (bb_std * 2)
    df['BB_LOWER'] = df['BB_MID'] - (bb_std * 2)
    # Bandwidth: mengukur squeeze
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID']

    # ---- 7. Stochastic RSI (14, 3, 3) ----
    rsi = df['RSI']
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)
    df['STOCH_RSI_K'] = stoch_rsi.rolling(window=3).mean() * 100
    df['STOCH_RSI_D'] = df['STOCH_RSI_K'].rolling(window=3).mean()

    # ---- 8. Swing Low (10 hari) untuk SL ----
    df['SWING_LOW_10'] = df['Low'].rolling(window=10).min()

    return df


# ============================================================
# WEEKLY CONFIRMATION (MULTI-TIMEFRAME)
# ============================================================
def check_weekly_uptrend(ticker):
    """
    Cek apakah saham dalam uptrend di timeframe weekly.
    Returns True jika weekly close > weekly SMA 20.
    """
    try:
        data = yf.download(ticker, period='6mo', interval='1wk', auto_adjust=True, progress=False)
        if data.empty or len(data) < 20:
            return False
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        weekly_close = data['Close'].iloc[-1]
        weekly_sma20 = data['Close'].rolling(window=20).mean().iloc[-1]
        
        if pd.isna(weekly_sma20):
            return False
        
        return weekly_close > weekly_sma20
    except Exception:
        return False


# ============================================================
# SCORING & PROCESSING
# ============================================================
def process_batch(tickers, market_regime='NEUTRAL'):
    """Memproses satu batch ticker dengan sistem skoring baru."""
    candidates = []
    score_threshold = SCORE_THRESHOLDS.get(market_regime, 6)
    failed_count = 0
    
    log.info(f"Mengunduh data untuk {len(tickers)} saham...")
    
    try:
        data = yf.download(tickers, period="1y", interval="1d", group_by='ticker', auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        log.error(f"Error download batch: {e}")
        return [], len(tickers)

    if len(tickers) == 1:
        data = {tickers[0]: data}

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data[ticker]
            else:
                try:
                    df = data[ticker].dropna(how='all')
                except KeyError:
                    failed_count += 1
                    continue
            
            if df.empty or len(df) < 60:
                continue

            # Hitung semua indikator
            df = calculate_technical(df)
            if df is None:
                continue

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            # ---- FILTER LIKUIDITAS ----
            avg_vol_20 = curr['VOL_MA20'] if pd.notna(curr['VOL_MA20']) else 0
            if curr['Close'] < MIN_PRICE or avg_vol_20 < MIN_VOLUME_AVG:
                continue

            # ---- SISTEM SKORING ----
            score = 0
            reasons = []

            # 1. RSI Oversold
            if pd.isna(curr['RSI']):
                continue
            
            if curr['RSI'] < 30:
                score += 3
                reasons.append(f"📉 RSI Oversold ({curr['RSI']:.1f})")
            elif 30 <= curr['RSI'] < 40:
                score += 1
                reasons.append(f"📉 RSI Murah ({curr['RSI']:.1f})")

            # 2. MACD Golden Cross / Menguat
            if 'MACDh_12_26_9' in df.columns:
                macd_h_now = curr['MACDh_12_26_9']
                macd_h_prev = prev['MACDh_12_26_9']
                
                if pd.notna(macd_h_now) and pd.notna(macd_h_prev):
                    if macd_h_now > 0 and macd_h_prev < 0:
                        score += 3
                        reasons.append("📈 MACD Golden Cross")
                    elif macd_h_now > macd_h_prev and macd_h_now > -0.5:
                        score += 1

            # 3. Volume Spike
            if pd.notna(curr['VOL_MA5']) and curr['VOL_MA5'] > 0:
                if curr['Volume'] > (curr['VOL_MA5'] * 1.5):
                    score += 2
                    vol_ratio = curr['Volume'] / curr['VOL_MA5']
                    reasons.append(f"📊 Volume Spike ({vol_ratio:.1f}x)")

            # 4. Trend Filter (SMA 200)
            if pd.notna(curr['SMA_200']) and curr['Close'] > curr['SMA_200']:
                score += 1
                reasons.append("⬆️ Uptrend (>MA200)")

            # 5. Bollinger Bands — Bounce dari Lower Band
            if pd.notna(curr['BB_LOWER']) and pd.notna(prev['BB_LOWER']):
                # Harga menyentuh/di bawah lower band kemarin, bounce hari ini
                if prev['Close'] <= prev['BB_LOWER'] * 1.01 and curr['Close'] > curr['BB_LOWER']:
                    score += 2
                    reasons.append("🔵 BB Lower Bounce")
                # Bollinger Squeeze (width < 5th percentile) + breakout
                elif pd.notna(curr['BB_WIDTH']):
                    bb_width_series = df['BB_WIDTH'].dropna()
                    if len(bb_width_series) >= 20:
                        width_pct5 = bb_width_series.quantile(0.05)
                        if prev['BB_WIDTH'] <= width_pct5 and curr['Close'] > curr['BB_MID']:
                            score += 2
                            reasons.append("🔵 BB Squeeze Breakout")

            # 6. Stochastic RSI — Oversold + Crossing Up
            if pd.notna(curr['STOCH_RSI_K']) and pd.notna(curr['STOCH_RSI_D']):
                if pd.notna(prev['STOCH_RSI_K']) and pd.notna(prev['STOCH_RSI_D']):
                    # K cross above D di zona oversold
                    if (curr['STOCH_RSI_K'] < 30 and 
                        prev['STOCH_RSI_K'] <= prev['STOCH_RSI_D'] and 
                        curr['STOCH_RSI_K'] > curr['STOCH_RSI_D']):
                        score += 2
                        reasons.append(f"🔄 StochRSI Cross Up ({curr['STOCH_RSI_K']:.0f})")

            # 7. OBV Rising (5 hari terakhir naik)
            if pd.notna(curr['OBV']) and pd.notna(curr['OBV_MA5']):
                if curr['OBV'] > curr['OBV_MA5']:
                    # Check OBV trend: last 5 days
                    obv_5d = df['OBV'].iloc[-5:]
                    if len(obv_5d) >= 5 and obv_5d.iloc[-1] > obv_5d.iloc[0]:
                        score += 1
                        reasons.append("📈 OBV Rising")

            # 8. Price above EMA 20
            if pd.notna(curr['EMA_20']) and curr['Close'] > curr['EMA_20']:
                score += 1
                reasons.append("⬆️ Above EMA20")

            # ---- TARGET PRICE (IMPROVED) ----
            atr = curr['ATR'] if pd.notna(curr['ATR']) else (curr['High'] - curr['Low'])
            
            # Stop Loss: yang lebih rendah antara swing low 10D atau ATR-based
            atr_sl = curr['Close'] - (atr * 1.5)
            swing_sl = curr['SWING_LOW_10'] if pd.notna(curr['SWING_LOW_10']) else atr_sl
            stop_loss = min(atr_sl, swing_sl)
            
            # Pastikan SL tidak negatif
            if stop_loss <= 0:
                stop_loss = curr['Close'] * 0.9  # Fallback: -10%
            
            # Risk = jarak ke SL
            risk = curr['Close'] - stop_loss
            if risk <= 0:
                risk = atr  # Fallback
            
            # Take Profit berdasarkan Fibonacci + ATR
            take_profit_1 = curr['Close'] + (risk * 1.618)   # TP1: Fib 1.618x risk
            take_profit_2 = curr['Close'] + (risk * 2.618)   # TP2: Fib 2.618x risk
            
            # Risk/Reward Ratio
            rr_ratio_1 = (take_profit_1 - curr['Close']) / risk if risk > 0 else 0
            rr_ratio_2 = (take_profit_2 - curr['Close']) / risk if risk > 0 else 0

            # ---- THRESHOLD CHECK ----
            if score >= score_threshold:
                candidates.append({
                    'symbol': ticker,
                    'price': curr['Close'],
                    'rsi': curr['RSI'],
                    'score': score,
                    'sl': stop_loss,
                    'tp1': take_profit_1,
                    'tp2': take_profit_2,
                    'rr1': rr_ratio_1,
                    'rr2': rr_ratio_2,
                    'reasons': reasons
                })

        except Exception as e:
            failed_count += 1
            log.debug(f"Error proses {ticker}: {e}")
            continue
            
    return candidates, failed_count


# ============================================================
# MAIN
# ============================================================
def main():
    start_time = time.time()
    log.info("=" * 60)
    log.info("STOCK SCREENER BOT - MULAI")
    log.info("=" * 60)
    
    # 1. Deteksi Market Regime
    market_regime, market_detail = get_market_regime()
    score_threshold = SCORE_THRESHOLDS.get(market_regime, 6)
    log.info(f"Market Regime: {market_regime} (Min Skor: {score_threshold})")
    
    # 2. Load Data Saham
    try:
        df_emiten = pd.read_csv('bei_universe.csv')
        all_tickers = df_emiten['symbol'].tolist()
        all_tickers = [x if x.endswith('.JK') else f"{x}.JK" for x in all_tickers]
    except FileNotFoundError:
        log.error("File bei_universe.csv tidak ditemukan!")
        send_error_alert("File bei_universe.csv tidak ditemukan!")
        return

    unique_tickers = list(set(all_tickers))
    log.info(f"Total saham: {len(unique_tickers)}")
    
    # 3. Proses per Batch
    final_candidates = []
    total_failed = 0
    
    total_batches = (len(unique_tickers) // BATCH_SIZE) + 1
    for i in range(0, len(unique_tickers), BATCH_SIZE):
        batch = unique_tickers[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log.info(f"Batch {batch_num}/{total_batches} ({len(batch)} saham)...")
        
        candidates, failed = process_batch(batch, market_regime)
        final_candidates.extend(candidates)
        total_failed += failed
        
        time.sleep(1)

    # 4. Multi-Timeframe Confirmation (Weekly)
    log.info(f"Cek konfirmasi weekly untuk {len(final_candidates)} kandidat...")
    for candidate in final_candidates:
        if check_weekly_uptrend(candidate['symbol']):
            candidate['score'] += 2
            candidate['reasons'].append("📅 Weekly Uptrend ✓")
    
    # 5. Urutkan & Filter Top Picks
    # Re-filter setelah weekly bonus (beberapa mungkin naik skor)
    final_candidates.sort(key=lambda x: x['score'], reverse=True)
    top_picks = final_candidates[:15]

    elapsed = int(time.time() - start_time)
    
    # 6. Kirim Laporan
    if top_picks:
        # Header
        regime_icon = {"BULLISH": "🟢", "NEUTRAL": "🟡", "BEARISH": "🔴"}.get(market_regime, "⚪")
        mode_text = "pandas-ta" if HAS_PANDAS_TA else "Manual"
        
        msg = f"🚀 *SCREENER SAHAM + SIGNAL* 🚀\n"
        msg += f"_{datetime.now().strftime('%d %b %Y %H:%M')}_\n\n"
        msg += f"{regime_icon} *Market: {market_regime}*\n"
        msg += f"_{market_detail}_\n"
        msg += f"_Min Skor: {score_threshold} | Scanned: {len(unique_tickers)} ({mode_text}) | ⏱ {elapsed}s_\n"
        
        if total_failed > 0:
            msg += f"_⚠️ {total_failed} saham gagal diproses_\n"
        msg += "\n"
        
        # Stock entries
        for i, stock in enumerate(top_picks, 1):
            icon = "🔥" if stock['score'] >= 8 else ("✅" if stock['score'] >= 6 else "💡")
            
            msg += f"{icon} *{stock['symbol'].replace('.JK', '')}* (Skor: {stock['score']})\n"
            msg += f"   💵 Close: Rp {stock['price']:,.0f}\n"
            msg += f"   🎯 TP1: Rp {stock['tp1']:,.0f} (RR {stock['rr1']:.1f}x)\n"
            msg += f"   🎯 TP2: Rp {stock['tp2']:,.0f} (RR {stock['rr2']:.1f}x)\n"
            msg += f"   🛡️ SL: Rp {stock['sl']:,.0f}\n"
            
            # Reasons — max 4 untuk readability
            reasons_display = stock['reasons'][:4]
            msg += f"   💡 `{'  '.join(reasons_display)}`\n"
            if len(stock['reasons']) > 4:
                msg += f"   _+{len(stock['reasons'])-4} sinyal lainnya_\n"
            msg += "\n"
        
        msg += f"⚠️ _Disclaimer: Bukan rekomendasi beli. DYOR!_"
            
        send_telegram(msg)
        log.info(f"Laporan terkirim ke Telegram. {len(top_picks)} picks ditemukan.")
    else:
        no_result_msg = (
            f"📊 *SCREENER SAHAM*\n"
            f"_{datetime.now().strftime('%d %b %Y %H:%M')}_\n\n"
            f"Tidak ada saham yang memenuhi kriteria hari ini.\n"
            f"Market: {market_regime} (Min Skor: {score_threshold})\n"
            f"_Scanned: {len(unique_tickers)} stocks in {elapsed}s_"
        )
        send_telegram(no_result_msg)
        log.info("Tidak ada kandidat hari ini.")
    
    log.info(f"Selesai dalam {elapsed} detik.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"BOT CRASH: {e}", exc_info=True)
        send_error_alert(str(e))
        sys.exit(1)
