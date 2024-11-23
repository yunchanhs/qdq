import time
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import tweepy
import random

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"
NEWS_API_KEY = "9288c1beaa4740f28223d9cca0e2af5a"
TWITTER_API_KEY = "FwlHdema5dEJR0XVVPcZ30UV5"
TWITTER_API_SECRET = "d841BV9P4bcBkn3ZgzwCColh5rsFNS8Tpvb3FURQi9nJGZD4TH"
TWITTER_ACCESS_TOKEN = "1559226950530781184-qLHOXkboV0ie4X7DlKE4Fuu7djGMT5"
TWITTER_ACCESS_TOKEN_SECRET = "0AllZylnl5HJvcTHcW4EFBzJxpY9R2ohS4tqnLDEdeAVl"

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

# 가격 예측을 위한 XGBoost 모델 학습
def predict_price(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    df['returns'] = df['close'].pct_change()
    df = df.dropna()

    # 피쳐와 타겟 설정
    X = df[['open', 'high', 'low', 'close', 'volume']]
    y = df['returns']

    # RandomForest 모델 학습 (XGBoost는 교체 가능)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    predicted_return = model.predict([X.iloc[-1]])  # 최근 데이터를 바탕으로 예측
    return predicted_return[0]

# 트레일링 스톱 (Trailing Stop)
def manage_trailing_stop(entry_price, current_price, ticker):
    trailing_stop_price = entry_price * 1.05  # 상승 시 5% 상승
    if current_price < trailing_stop_price:
        # 트레일링 스톱에 도달하면 매도
        coin_balance = get_balance(ticker.split('-')[1])
        sell_result = sell_crypto_currency(ticker, coin_balance)
        if sell_result:
            print(f"[{ticker}] 트레일링 스톱 발동, 매도 완료: {coin_balance} 개")
            return True
    return False

# 뉴스 감성 분석
def analyze_news_sentiment():
    # 예시로 사용, 실제로는 암호화폐 관련 뉴스 API를 호출하거나 트위터 피드를 분석할 수 있음
    news_url = f'https://newsapi.org/v2/everything?q=crypto&apiKey={NEWS_API_KEY}'
    response = requests.get(news_url)
    news_data = response.json()
    sentiment = 0
    for article in news_data['articles']:
        sentiment += TextBlob(article['title']).sentiment.polarity
    return sentiment

# 소셜 미디어 분석 (트위터 예시)
def analyze_twitter_sentiment():
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    
    # 트위터에서 암호화폐 관련 트윗을 검색
    tweets = api.search(q="cryptocurrency", count=100, lang="ko")
    sentiment = 0
    for tweet in tweets:
        sentiment += TextBlob(tweet.text).sentiment.polarity
    return sentiment

# 메인 로직
if __name__ == "__main__":
    print("자동매매 시작!")
    try:
        while True:
            tickers = pyupbit.get_tickers(fiat="KRW")
            krw_balance = get_balance("KRW")
            
            # 실시간 뉴스 분석
            news_sentiment = analyze_news_sentiment()
            # 실시간 트위터 분석
            twitter_sentiment = analyze_twitter_sentiment()

            for ticker in tickers:
                try:
                    now = datetime.now()

                    # 쿨다운 타임 체크
                    if ticker in recent_trades and now - recent_trades[ticker] < COOLDOWN_TIME:
                        continue
                    
                    # 각 지표 계산
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    predicted_return = predict_price(ticker)
                    current_price = pyupbit.get_current_price(ticker)
                    
                    # 매수 조건 (MACD 크로스, RSI, 예측 가격 상승)
                    if macd > signal and rsi < 30 and predicted_return > 0.02 and krw_balance > 5000 and (news_sentiment > 0 or twitter_sentiment > 0):
                        buy_amount = krw_balance * 0.05  # 잔고의 5% 매수
                        buy_result = buy_crypto_currency(ticker, buy_amount)
                        if buy_result:
                            entry_prices[ticker] = current_price  # 진입가 저장
                            recent_trades[ticker] = now
                            print(f"[{ticker}] 매수 완료. 금액: {buy_amount:.2f}, 가격: {current_price:.2f}")
                    
                    # 매도 조건 (손절/익절 또는 트레일링 스톱)
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

                        # 트레일링 스톱 관리
                        if manage_trailing_stop(entry_price, current_price, ticker):
                            continue

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

            # 1분 대기
            time.sleep(60)

    except Exception as e:
        print(f"시스템 에러: {e}")
