import time
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"
NEWS_API_KEY = "9288c1beaa4740f28223d9cca0e2af5a"

# 손절 및 익절 비율 설정
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절

# 쿨다운 타임 설정
COOLDOWN_TIME = timedelta(minutes=5)

# 최근 매매 기록 저장 (쿨다운 타임 관리)
recent_trades = {}

# 진입가 저장
entry_prices = {}

# Upbit API 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 매수 함수
def buy_crypto_currency(ticker, amount):
    try:
        result = upbit.buy_market_order(ticker, amount)
        return result
    except Exception as e:
        print(f"[{ticker}] 매수 에러: {e}")
        return None

# 매도 함수
def sell_crypto_currency(ticker, amount):
    try:
        result = upbit.sell_market_order(ticker, amount)
        return result
    except Exception as e:
        print(f"[{ticker}] 매도 에러: {e}")
        return None

# 잔고 확인 함수
def get_balance(ticker):
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            return float(b.get('balance', 0))
    return 0

# MACD 계산 함수
def get_macd(ticker, short_window=12, long_window=26, signal_window=9):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    short_ema = df['close'].ewm(span=short_window).mean()
    long_ema = df['close'].ewm(span=long_window).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window).mean()
    return macd.iloc[-1], signal.iloc[-1]

# RSI 계산 함수
def get_rsi(ticker, period=14):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# Bollinger Bands 계산 함수
def get_bollinger_bands(ticker, window=20, num_std=2):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band.iloc[-1], lower_band.iloc[-1], sma.iloc[-1]

# Moving Average 계산 함수
def get_moving_average(ticker, window=50):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    sma = df['close'].rolling(window=window).mean()
    return sma.iloc[-1]

# 뉴스 감성 분석
def analyze_news_sentiment():
    news_url = f'https://newsapi.org/v2/everything?q=crypto&apiKey={NEWS_API_KEY}'
    response = requests.get(news_url)
    news_data = response.json()
    sentiment = 0
    for article in news_data['articles']:
        sentiment += TextBlob(article['title']).sentiment.polarity
    return sentiment

# 메인 로직
if __name__ == "__main__":
    print("자동매매 시작!")
    try:
        last_news_time = datetime.min  # 뉴스 API 호출 시간 초기화
        while True:
            tickers = pyupbit.get_tickers(fiat="KRW")
            krw_balance = get_balance("KRW")
            
            # 실시간 뉴스 분석 (5분에 한 번만 호출)
            if datetime.now() - last_news_time > timedelta(minutes=5):
                news_sentiment = analyze_news_sentiment()
                last_news_time = datetime.now()  # 뉴스 호출 시간 업데이트
            else:
                news_sentiment = 0  # 뉴스 분석이 호출되지 않도록 설정

            for ticker in tickers:
                try:
                    now = datetime.now()

                    # 쿨다운 타임 체크
                    if ticker in recent_trades and now - recent_trades[ticker] < COOLDOWN_TIME:
                        continue
                    
                    # 각 지표 계산
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    upper_band, lower_band, sma = get_bollinger_bands(ticker)
                    moving_avg = get_moving_average(ticker)
                    current_price = pyupbit.get_current_price(ticker)

                    # 매수 조건 (MACD 크로스, RSI, Bollinger Band 하단 근접, Moving Average 상승 등)
                    if macd > signal and rsi < 30 and current_price < lower_band and moving_avg > current_price and krw_balance > 5000 and news_sentiment > 0:
                        buy_amount = krw_balance * 0.05  # 잔고의 5% 매수
                        buy_result = buy_crypto_currency(ticker, buy_amount)
                        if buy_result:
                            entry_prices[ticker] = current_price  # 진입가 저장
                            recent_trades[ticker] = now
                            print(f"[{ticker}] 매수 완료. 금액: {buy_amount:.2f}, 가격: {current_price:.2f}")
                    
                    # 매도 조건 (손절/익절)
                    if ticker in entry_prices:
                        entry_price = entry_prices[ticker]
                        change_ratio = (current_price - entry_price) / entry_price

                        # 손절 또는 익절
                        if change_ratio <= STOP_LOSS_THRESHOLD or change_ratio >= TAKE_PROFIT_THRESHOLD:
                            coin_balance = get_balance(ticker.split('-')[1])
                            sell_result = sell_crypto_currency(ticker, coin_balance)
                            if sell_result:
                                recent_trades[ticker] = now
                                print(f"[{ticker}] 매도 완료. 잔고: {coin_balance:.4f}, 가격: {current_price:.2f}")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

            # 1분 대기
            time.sleep(60)

    except Exception as e:
        print(f"시스템 에러: {e}")
