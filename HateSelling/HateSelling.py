import pyupbit
import requests
import time
from Config import access, secret

upbit = pyupbit.Upbit(access, secret)

def get_orderbook_and_price():
    orderbook_url = "https://api.upbit.com/v1/orderbook?markets=KRW-BTC&level=0"
    ticker_url = "https://api.upbit.com/v1/ticker"
    orderbook = requests.get(orderbook_url).json()[0]
    price = requests.get(ticker_url, params={"markets": "KRW-BTC"}).json()[0]['trade_price']
    return orderbook['orderbook_units'], price

def is_order_filled(uuid):
    if uuid is None:
        return False

    for _ in range(100):  # 최대 10초간 체결 대기
        order = upbit.get_order(uuid)
        trades = order.get('trades', [])
        if trades:
            return True
        time.sleep(0.1)
    return False

def RunHateSell():
    print("혐사 시작")
    while True:
        try:
            krw_balance = upbit.get_balance("KRW") - 5000
            if krw_balance < 5000:
                print('잔액 부족, 매매 종료')
                break
            units, _ = get_orderbook_and_price()
            units_ask = sorted(units, key=lambda x: x["ask_price"])[0]
            units_bid = sorted(units, key=lambda x: x["bid_price"], reverse=True)[0]

            if units_ask['ask_price'] <= units_bid['bid_price']:

                krw = int(units_ask['ask_price'] * min(units_ask['ask_size'], units_bid['bid_size']))

                if krw < 5000:
                    continue

                buy_result = upbit.buy_market_order("KRW-BTC", min(krw, krw_balance))
                uuid = buy_result['uuid']

                if is_order_filled(uuid):
                    print(f'{krw} 매수 완료')
                    btc_balance = upbit.get_balance("BTC")
                    upbit.sell_limit_order("KRW-BTC",
                                           units_bid['bid_price'], btc_balance)
                    print(f'매도 완료')

            time.sleep(1)

        except Exception as e:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), e)
            time.sleep(3)