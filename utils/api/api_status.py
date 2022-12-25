from urllib import response
from api.creds import client_connect
import krakenex
from tda import client 
from pykrakenapi import KrakenAPI

print('      API      STATUS')

# connect to TDA API
try:
    tda_client = client_connect('TDA', 'private/creds.ini')

    response = tda_client.get_price_history('AAPL',
        period_type=client.Client.PriceHistory.PeriodType.YEAR,
        period=client.Client.PriceHistory.Period.ONE_YEAR,
        frequency_type=client.Client.PriceHistory.FrequencyType.DAILY,
        frequency=client.Client.PriceHistory.Frequency.DAILY)
    assert response.status_code == 200, response.raise_for_status()
    print('TD Ameritrade:   UP')
except:
    print('TD Ameritrade:  DOWN')

# connect to Coinbase API
try:
    cb_client = client_connect('CB', 'private/creds.ini')
    print('Coinbase:        UP')
except:
    print('Coinbase:       DOWN')

try:
    api = krakenex.API()
    k = KrakenAPI(api)
    ohlc, last = k.get_ohlc_data("BCHUSD")
    print('Kraken:          UP')
except:
    print('Kraken:         DOWN')