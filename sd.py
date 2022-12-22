import datetime
import logging
import pyupbit
from time import sleep
import pandas
import time

logging.basicConfig(filename="trade_log.txt", level=logging.INFO)

access = "UxavXwLQLeMi6iEjxb4p8Dy6rwmk9GhzB2l8Dr8I"
secret = "2NPZtGBJ0VV9sPjcvL76kd6N4opwVgpxpj1jUi3E"
server_url = "https://api.upbit.com/v1/market/all"

upbit = pyupbit.Upbit(access, secret)
print("autotrade start")

class BitBot(pyupbit.Upbit):

    def __init__(self, api_key): # api_key : dict 형태 --> api_key['access', 'secret']
        super().__init__(api_key['access'], api_key['secret'])
        self.ticker_list = pyupbit.get_tickers(fiat="KRW")  # 거래가능한 KRW 코인 추출
        self.account_info = self.get_balances()
        self.monitoring_list = dict()

        for ticker in self.ticker_list:
            self.monitoring_list[ticker] = 0

        """ 구조 재조정 필요
        self.monitoring_list = dict()
        
        monitoring_list 구조
            코인심볼명(key) : 피라미딩레벨(value), 현재매수가(value),
                            다음목표매수가(value), 손절가(value) 
        """

    def get_monitoring_list(self):
        return self.monitoring_list

    def set_monitoring_list(self, ticker, price, gubun=0):  # 0:add 1:edit 2:del
        pass

    def get_current_ohlc(self, ticker):
        data = pyupbit.get_daily_ohlcv_from_base(ticker, base=9).iloc[-1]
        return data['open'], data['high'], data['low'], data['close']

    def get_order_book(self, ticker, position, slippage=2):
        # ticker : 코인심볼명, postition : 매매방향, slippage : 호가레벨최대허용치
        # 슬리피지 기본값은 최대 2틱

        orderbook_list = pyupbit.get_orderbook(ticker)
        bids_asks = orderbook_list[0]['orderbook_units']

        if position == "BUY":
            return float(bids_asks[slippage-1]['ask_price'])

        elif position == "SELL":
            return float(bids_asks[slippage-1]['bid_price'])

def buy_order(self, ticker, order_price, maximum_price):
        quant_size = maximum_price / order_price
        # msg = ticker, order_price, quant_size, " BUY"
        ret = self.buy_limit_order(ticker, order_price, quant_size)
        return ret

def sell_order(self, ticker, order_price):
        quant_size = 0

        for asset in self.get_balances():  # 가지고 있는 자산 조회
            if asset['currency'] == ticker[4:]:
                quant_size = float(asset['balance'])

        if quant_size > 0:
            msg = ticker, order_price, quant_size, " SELL"
            ret = self.sell_limit_order(ticker, order_price, quant_size)
            return ret

def run(self):
        my_krw = self.get_balance("KRW")

        for ticker in self.ticker_list:  # 거래가능한 모든 코인리스트 만큼 반복

            flag = self.scan(ticker)

            if flag == "BUY":
                best_price = self.get_order_book(ticker, "BUY")  # 슬리피지 적용한 매수가
                self.buy_order(ticker, best_price, my_krw)  # 심볼, 주문적정가, 보유금액(주문가능최대금액)

            elif flag == "SELL":
                best_price = self.get_order_book(ticker, "SELL")  # 슬리피지 적용한 매도가
                self.sell_order(ticker, best_price)
              

def rsi(ohlc: pandas.DataFrame, period: int = 14):
    delta = ohlc["close"].diff()
    ups, downs = delta.copy(), delta.copy()
    ups[ups < 0] = 0
    downs[downs > 0] = 0

    AU = ups.ewm(com = period-1, min_periods = period).mean()
    AD = downs.abs().ewm(com = period-1, min_periods = period).mean()
    RS = AU/AD

    return pandas.Series(100 - (100/(1 + RS)), name = "RSI") 

coinlist = ["KRW"]
lower28 = []
higher70 = []

def scan(self, ticker):
    for i in range(len(coinlist)):
     lower28.append(False)
     higher70.append(False)
     open, high, low, close = self.get_current_ohlc(ticker)
     data = pyupbit.get_ohlcv(ticker=coinlist[i], interval="minute3")
     now_rsi = rsi(data, 14).iloc[-1]
     open_high_rate = float((high - open) / open)    # 시가에서 고가 변동률
     open_close_rate = float((close - open) / open)  # 시가에서 종가 변동률
     close_high_rate = float((high - close) / close)
     corrected_value = self.monitoring_list[ticker] * 0.01
     msg = f"{ticker} 시고:{open_high_rate:.3f}, 시종:{open_close_rate:.3f}, 종고:{close_high_rate:.3f}{open, high, low, close}"
     condition_1_1 = (0.05 + corrected_value < open_close_rate < 0.06 + corrected_value)
     condition_1_2 = close_high_rate < 0.05
     condition_2_1 = (self.monitoring_list[ticker] > 0)
     condition_2_2 = (open_high_rate > 0.10 and close_high_rate < -0.05)
     condition_2_3 = open_close_rate < 0.04
     print("코인명: ", coinlist[i])
     print("현재시간: ", datetime.datetime.now())
     print("RSI :", now_rsi)

     if condition_1_1 and condition_1_2 and now_rsi <= 28 : 
            
        lower28[i] = True
     elif now_rsi >= 33 and lower28[i] == True:
      
        lower28[i] = False
     elif now_rsi >= 70 and higher70[i] and condition_2_1 and (condition_2_2 or condition_2_3) == False:
                
        higher70[i] = True
     elif now_rsi <= 60 :
            
        higher70[i] = False
    
    time.sleep(1)
    
def buy(coin):
       money = upbit.get_balance("KRW")
       if money < 20000 :
          res = upbit.buy_market_order(coin, money)
       elif money < 50000:
          res = upbit.buy_market_order(coin, money*0.4)
       elif money < 100000 :
          res = upbit.buy_market_order(coin, money*0.3)
       else :
          res = upbit.buy_market_order(coin, money*0.2)
          return

def sell(coin):
       amount = upbit.get_balance(coin)
       cur_price = pyupbit.get_current_price(coin)
       total = amount * cur_price
       if total < 20000 :
          res = upbit.sell_market_order(coin, amount)
       elif total < 50000:
          res = upbit.sell_market_order(coin, amount*0.4)
       elif total < 100000:
          res = upbit.sell_market_order(coin, amount*0.3)        
       else :
          res = upbit.sell_market_order(coin, amount*0.2)
          return
time.sleep(1)
