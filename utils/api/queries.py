from tda.client.synchronous import Client as TDA_Client
from coinbase.wallet.client import Client as CB_Client
from polygon import RESTClient as POLY_CLIENT
import pandas as pd
from pandas import DataFrame
import json
import talib as ta
import datetime as dt
from datetime import datetime, timedelta
import numpy as np
from typing import Union
from tqdm import tqdm
import multiprocessing as mp
import os
from utils.api.connections import client_connect

def get_data(client : Union[TDA_Client, CB_Client, POLY_CLIENT ], features : dict, api : str, symbol: str, normalize=True, save=False, save_path='', log=False, **kwargs):
    if isinstance(client, TDA_Client):
        all_frequencies = {
            'ONE_MIN': client.PriceHistory.Frequency.EVERY_MINUTE,
            'FIVE_MIN': client.PriceHistory.Frequency.EVERY_FIVE_MINUTES,
            'TEN_MIN': client.PriceHistory.Frequency.EVERY_TEN_MINUTES,
            'FIFTEEN_MIN': client.PriceHistory.Frequency.EVERY_FIFTEEN_MINUTES,
            'THIRTY_MIN': client.PriceHistory.Frequency.EVERY_THIRTY_MINUTES,
            'DAILY': client.PriceHistory.Frequency.DAILY,
            'WEEKLY': client.PriceHistory.Frequency.WEEKLY,
            'MONTHLY': client.PriceHistory.Frequency.MONTHLY
        }
        all_frequency_types = {
            'MINUTE': client.PriceHistory.FrequencyType.MINUTE,
            'DAILY': client.PriceHistory.FrequencyType.DAILY,
            'WEEKLY': client.PriceHistory.FrequencyType.WEEKLY,
            'MONTHLY': client.PriceHistory.FrequencyType.MONTHLY
        }
        all_periods = {
            'ONE_YEAR': client.PriceHistory.Period.ONE_YEAR,
            'TWO_YEAR': client.PriceHistory.Period.TWO_YEARS,
            'THREE_YEAR': client.PriceHistory.Period.THREE_YEARS,
            'FIVE_YEAR': client.PriceHistory.Period.FIVE_YEARS,
            'TEN_YEAR': client.PriceHistory.Period.TEN_YEARS,
            'FIFTEEN_YEAR': client.PriceHistory.Period.FIFTEEN_YEARS,
            'YEAR_TO_DATE': client.PriceHistory.Period.YEAR_TO_DATE,
            'ONE_MONTH': client.PriceHistory.Period.ONE_MONTH,
            'TWO_MONTH': client.PriceHistory.Period.TWO_MONTHS,
            'THREE_MONTH': client.PriceHistory.Period.THREE_MONTHS,
            'SIX_MONTH': client.PriceHistory.Period.SIX_MONTHS,
            'ONE_DAY': client.PriceHistory.Period.ONE_DAY,
            'TWO_DAY': client.PriceHistory.Period.TWO_DAYS,
            'THREE_DAY': client.PriceHistory.Period.THREE_DAYS,
            'FOUR_DAY': client.PriceHistory.Period.FOUR_DAYS,
            'FIVE_DAY': client.PriceHistory.Period.FIVE_DAYS
        }
        all_period_types = {
            'DAY': client.PriceHistory.PeriodType.DAY,
            'MONTH': client.PriceHistory.PeriodType.MONTH,
            'YEAR': client.PriceHistory.PeriodType.YEAR,
            'YEAR_TO_DATE': client.PriceHistory.PeriodType.YEAR_TO_DATE
        }
        params = {
            'symbol': symbol, 
            'period_type': all_period_types[kwargs['period_type']], 
            'period': all_periods[kwargs['period']], 
            'frequency_type': all_frequency_types[kwargs['frequency_type']], 
            'frequency': all_frequencies[kwargs['frequency']],
            'need_extended_hours_data': 'true'
        }
        payload = { key: val for key, val in params.items() if val != None } 
        data = json.dumps(client.get_price_history(**payload).json(), indent=4)
    else:
        data = client.stocks_equities_aggregates(symbol, **kwargs)

    '''Extract Features'''
    if isinstance(client, TDA_Client):
        data = json.loads(data)
        symbol = data['symbol']
    else:
        data = data.__dict__
        symbol = data['ticker']
    if isinstance(client, TDA_Client):
        vars = {
            'open': np.array([ candle['open'] for candle in data['candles'] ]),
            'close': np.array([ candle['close'] for candle in data['candles'] ]),
            'high': np.array([ candle['high'] for candle in data['candles'] ]),
            'low': np.array([ candle['low'] for candle in data['candles'] ]),
            'volume': np.array([ candle['volume'] for candle in data['candles'] ]),
            'datetime': np.array([ pd.to_datetime(candle['datetime'], unit='ms') for candle in data['candles'] ])
        }
    elif isinstance(client, POLY_CLIENT):
        vars = {
            'open': np.array([ candle['o'] for candle in data['results'] ]),
            'close': np.array([ candle['c'] for candle in data['results'] ]),
            'high': np.array([ candle['h'] for candle in data['results'] ]),
            'low': np.array([ candle['l'] for candle in data['results'] ]),
            'volume': np.array([ candle['v'] for candle in data['results'] ]),
            'datetime': np.array([ pd.to_datetime(candle['t'], unit='ms') for candle in data['results'] ])
        }
    
    start_date = vars['datetime'][0]
    end_date = vars['datetime'][-1]
    vars['volume'] = vars['volume'].astype(float)

    # use ta-lib to calculate features given as input 
    for func, params in features.items():
        if func in ta.get_functions():
            # ta-lib overlap studies
            if func == 'BBANDS':
                upperband, middleband, lowerband = ta.BBANDS(vars['close'], **params)
                vars['UPPER_BBAND'] = upperband
                vars['MIDDLE_BBAND'] = middleband
                vars['LOWER_BBAND'] = lowerband
            elif func == 'DEMA':
                vars['DEMA'] = ta.DEMA(vars['close'], **params)
            elif func == 'EMA':
                vars['EMA'] = ta.EMA(vars['close'], **params)
            elif func == 'HT_TRENDLINE':
                vars['HT_TRENDLINE'] = ta.HT_TRENDLINE(vars['close'], **params)
            elif func == 'KAMA':
                vars['KAMA'] = ta.KAMA(vars['close'], **params)
            elif func == 'MA':
                vars['MA'] = ta.MA(vars['close'], **params)
            elif func == 'MAMA':
                vars['MAMA'], vars['FAMA'] = ta.MAMA(vars['close'], **params)
            # elif func == 'MAVP':
            #     vars['mavp_real'] = getattr(ta, 'MAVP')(vars['close'], **params)
            elif func == 'MIDPOINT':
                vars['MIDPOINT'] = ta.MIDPOINT(vars['close'], **params)
            elif func == 'MIDPRICE':
                vars['MIDPRICE'] = ta.MIDPRICE(vars['high'], vars['low'], **params)
            elif func == 'SAR':
                vars['SAR'] = ta.SAR(vars['high'], vars['low'], **params)
            elif func == 'SAREXT':
                vars['SAREXT'] = ta.SAREXT(vars['high'], vars['low'], **params)
            elif func == 'SMA':
                vars['SMA'] = ta.SMA(vars['close'], **params)
            elif func == 'T3':
                vars['T3'] = ta.T3(vars['close'], **params)
            elif func == 'TEMA':
                vars['TEMA'] = ta.TEMA(vars['close'], **params)
            elif func == 'TRIMA':
                 vars['TRIMA'] = ta.TRIMA(vars['close'], **params)
            elif func == 'WMA':
                vars['WMA'] = ta.WMA(vars['close'], **params)
            #  ta-lib momentum indicators 
            elif func == 'ADX': 
                vars['ADX'] = ta.ADX(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'ADXR':
                vars['ADXR'] = ta.ADXR(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'APO':
                vars['APO'] = ta.APO(vars['close'], **params)
            elif func == 'AROON':
                vars['AROONDOWN'], vars['AROONUP'] = getattr(ta, 'AROON')(vars['high'], vars['low'], **params)
            elif func == 'AROONOSC':
                vars['AROONOSC'] = getattr(ta, 'AROONOSC')(vars['high'], vars['low'], **params)
            elif func == 'BOP':
                vars['BOP'] = getattr(ta, 'BOP')(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CCI':
                vars['CCI'] = getattr(ta, 'CCI')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CMO':
                vars['CMO'] = getattr(ta, 'CMO')(vars['close'], **params)
            elif func == 'DX':
                vars['DX'] = getattr(ta, 'DX')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'MACD':
                vars['MACD'], vars['MACD_SIGNAL'], vars['MACD_HIST'] = getattr(ta, 'MACD')(vars['close'], **params)
            elif func == 'MACDEXT':
                vars['MACDEXT'], vars['MACDEXT_SIGNAL'], vars['MACDEXT_EXT'] = getattr(ta, 'MACDEXT')(vars['close'], **params)
            elif func == 'MACDFIX':
                vars['MACDFIX'], vars['MACDFIX_SIGNAL'], vars['MACDFIX_EXT'] = getattr(ta, 'MACDFIX')(vars['close'], **params)
            elif func == 'MFI':
                vars['MFI'] = getattr(ta, 'MFI')(vars['high'], vars['low'], vars['close'], vars['volume'], **params)
            elif func == 'MINUS_DI':
                vars['MINUS_DI'] = getattr(ta, 'MINUS_DI')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'MINUS_DM':
                vars['MINUS_DM'] = getattr(ta, 'MINUS_DM')(vars['high'], vars['low'], **params)
            elif func == 'MOM':
                vars['MOM'] = getattr(ta, 'MOM')(vars['close'], **params)
            elif func == 'PLUS_DI':
                vars['PLUS_DI'] = getattr(ta, 'PLUS_DI')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'PLUS_DM':
                vars['PLUS_DM'] = getattr(ta, 'PLUS_DM')(vars['high'], vars['low'], **params)
            elif func == 'PPO':
                vars['PPO'] = getattr(ta, 'PPO')(vars['close'], **params)
            elif func == 'ROC':
                vars['ROC'] = getattr(ta, 'ROC')(vars['close'], **params)
            elif func == 'ROCP':
                vars['ROCP'] = getattr(ta, 'ROCP')(vars['close'], **params)
            elif func == 'ROCR100':
                vars['ROCR100'] = getattr(ta, 'ROCR100')(vars['close'], **params)
            elif func == 'RSI':
                vars['RSI'] = getattr(ta, 'RSI')(vars['close'], **params)
            elif func == 'STOCH':
                vars['STOCH_SLOWK'], vars['STOCH_SLOWD'] = getattr(ta, 'STOCH')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'STOCHF':
                vars['STOCH_FASTK'], vars['STOCH_FASTD'] = getattr(ta, 'STOCHF')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'STOCHRSI':
                vars['STOCHRSI_FASTK'], vars['STOCHRSI_FASTD'] = getattr(ta, 'STOCHRSI')(vars['close'], **params)
            elif func == 'TRIX':
                vars['TRIX'] = getattr(ta, 'TRIX')(vars['close'], **params)
            elif func == 'ULTOSC':
                vars['ULTOSC'] = getattr(ta, 'ULTOSC')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'WILLR':
                vars['WILLR'] = getattr(ta, 'WILLR')(vars['high'], vars['low'], vars['close'], **params)
            # ta-lib volume indicators 
            elif func == 'AD':
                vars['AD'] = getattr(ta, 'AD')(vars['high'], vars['low'], vars['close'], vars['volume'])
            elif func == 'ADOSC':
                vars['ADOSC'] = getattr(ta, 'ADOSC')(vars['high'], vars['low'], vars['close'], vars['volume'], **params)
            elif func == 'OBV':
                vars['OBV'] = getattr(ta, 'OBV')(vars['close'], vars['volume'])
            # ta-lib volatility indicators 
            elif func == 'ATR':
                vars['ATR'] = getattr(ta, 'ATR')(vars['high'], vars['low'], vars['close'], **params)
            elif func == 'TRANGE':
                vars['TRANGE'] = getattr(ta, 'TRANGE')(vars['high'], vars['low'], vars['close'])
            # ta-lib price transform
            elif func == 'AVGPRICE':
                vars['AVGPRICE'] = getattr(ta, 'AVGPRICE')(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'MEDPRICE':
                vars['MEDPRICE'] = getattr(ta, 'MEDPRICE')(vars['high'], vars['low'])
            elif func == 'TYPPRICE':
                vars['TYPPRICE'] = getattr(ta, 'TYPPRICE')(vars['high'], vars['low'], vars['close'])
            elif func == 'WCLPRICE':
                vars['WCLPRICE'] = getattr(ta, 'WCLPRICE')(vars['high'], vars['low'], vars['close'])
            # ta-lib cycle indicators 
            elif func == 'HT_DCPERIOD':
                vars['HT_DCPERIOD'] = ta.HT_DCPERIOD(vars['close'])
            elif func == 'HT_DCPHASE':
                vars['HT_DCPHASE'] = ta.HT_DCPHASE(vars['close'])
            elif func == 'HT_PHASOR':
                vars['HT_PHASOR_INPHASE'], vars['HT_PHASOR_QUADRATURE'] = ta.HT_PHASOR(vars['close'])
            elif func == 'HT_SINE':
                vars['HT_SINE'], vars['HT_LEADSINE'] = ta.HT_SINE(vars['close'])
            elif func == 'HT_TRENDMODE':
                vars['HT_TRENDMODE'] = ta.HT_TRENDMODE(vars['close'])
            # ta-lib pattern recognition
            elif func == 'CDL2CROWS':
                vars['CDL2CROWS'] = ta.CDL2CROWS(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDL3BLACKCROWS':
                vars['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDL3INSIDE':
                vars['CDL3INSIDE'] = ta.CDL3INSIDE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDL3LINESTRIKE':
                vars['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDL3OUTSIDE':
                vars['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDL3STARSINSOUTH':
                vars['CDL3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDL3WHITESOLDIERS':
                vars['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLABANDONEDBABY':
                vars['CDLABANDONEDBABY'] = ta.CDLABANDONEDBABY(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CDLADVANCEBLOCK':
                vars['CDLADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLBELTHOLD':
                vars['CDLBELTHOLD'] = ta.CDLBELTHOLD(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLBREAKAWAY':
                vars['CDLBREAKAWAY'] = ta.CDLBREAKAWAY(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLCLOSINGMARUBOZU':
                vars['CDLCLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLCONCEALBABYSWALL':
                vars['CDLCONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLCOUNTERATTACK':
                vars['CDLCOUNTERATTACK'] = ta.CDLCOUNTERATTACK(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLDARKCLOUDCOVER':
                vars['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CDLDOJI':
                vars['CDLDOJI'] = ta.CDLDOJI(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLDOJISTAR':
                vars['CDLDOJISTAR'] = ta.CDLDOJISTAR(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLDRAGONFLYDOJI':
                vars['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLENGULFING':
                vars['CDLENGULFING'] = ta.CDLENGULFING(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLEVENINGDOJISTAR':
                vars['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CDLEVENINGSTAR':
                vars['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CDLGAPSIDESIDEWHITE':
                vars['CDLGAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLGRAVESTONEDOJI':
                vars['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHAMMER':
                vars['CDLHAMMER'] = ta.CDLHAMMER(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHANGINGMAN':
                vars['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHARAMI':
                vars['CDLHARAMI'] = ta.CDLHARAMI(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHARAMICROSS':
                vars['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHIGHWAVE':
                vars['CDLHIGHWAVE'] = ta.CDLHIGHWAVE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHIKKAKE':
                vars['CDLHIKKAKE'] = ta.CDLHIKKAKE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHIKKAKEMOD':
                vars['CDLHIKKAKEMOD'] = ta.CDLHIKKAKEMOD(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLHOMINGPIGEON':
                vars['CDLHOMINGPIGEON'] = ta.CDLHOMINGPIGEON(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLIDENTICAL3CROWS':
                vars['CDLIDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLINNECK':
                vars['CDLINNECK'] = ta.CDLINNECK(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLINVERTEDHAMMER':
                vars['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLKICKING':
                vars['CDLKICKING'] = ta.CDLKICKING(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLKICKINGBYLENGTH':
                vars['CDLLADDERBOTTOM'] = ta.CDLLADDERBOTTOM(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLLONGLEGGEDDOJI':
                vars['CDLLONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLLONGLINE':
                vars['CDLLONGLINE'] = ta.CDLLONGLINE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLMARUBOZU':
                vars['CDLMARUBOZU'] = ta.CDLMARUBOZU(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLMATCHINGLOW':
                vars['CDLMATCHINGLOW'] = ta.CDLMATCHINGLOW(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLMATHOLD':
                vars['CDLMATHOLD'] = ta.CDLMATHOLD(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CDLMORNINGDOJISTAR':
                vars['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CDLMORNINGSTAR':
                vars['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(vars['open'], vars['high'], vars['low'], vars['close'], **params)
            elif func == 'CDLONNECK':
                vars['CDLONNECK'] = ta.CDLONNECK(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLPIERCING':
                vars['CDLPIERCING'] = ta.CDLPIERCING(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLRICKSHAWMAN':
                vars['CDLRICKSHAWMAN'] = ta.CDLRICKSHAWMAN(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLRISEFALL3METHODS':
                vars['CDLRISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLSEPARATINGLINES':
                vars['CDLSEPARATINGLINES'] = ta.CDLSEPARATINGLINES(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLSHOOTINGSTAR':
                vars['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLSHORTLINE':
                vars['CDLSHORTLINE'] = ta.CDLSHORTLINE(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLSPINNINGTOP':
                vars['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLSTALLEDPATTERN':
                vars['CDLSTALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLSTICKSANDWICH':
                vars['CDLSTICKSANDWICH'] = ta.CDLSTICKSANDWICH(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLTAKURI':
                vars['CDLTAKURI'] = ta.CDLTAKURI(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLTASUKIGAP':
                vars['CDLTASUKIGAP'] = ta.CDLTASUKIGAP(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLTHRUSTING':
                vars['CDLTHRUSTING'] = ta.CDLTHRUSTING(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLTRISTAR':
                vars['CDLTRISTAR'] = ta.CDLTRISTARI(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLUNIQUE3RIVER':
                vars['CDLUNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLUPSIDEGAP2CROWS':
                vars['CDLUPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(vars['open'], vars['high'], vars['low'], vars['close'])
            elif func == 'CDLXSIDEGAP3METHODS':
                vars['CDLXSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(vars['open'], vars['high'], vars['low'], vars['close'])
        elif func == '%B':
            if 'UPPER_BBAND' not in vars.keys():
                upperband, middleband, lowerband = ta.BBANDS(vars['close'], **params)
                vars['%B'] = (vars['close'] - lowerband) / (upperband - lowerband)
            else: 
                vars['%B'] = (vars['close'] - vars['LOWER_BBAND']) / (vars['UPPER_BBAND'] - vars['LOWER_BBAND'])
        elif func == 'VIX':
            fred_client = client_connect('FRED', 'private/creds.ini')
            temp = fred_client.get_series('VIXCLS')
            vix = []
            for date in vars['datetime']:
                try:
                    vix.append(temp[f'{date:%Y}-{date:%m}-{date:%d}'])
                except:
                    vix.append(None)
            vars['VIX'] = vix


    df = DataFrame.from_dict(vars)
    df = df.dropna()            # this often drops some rows since the time range given to the technical indicators go past the data.
                                # you still get most of the data you ask for for the most part, but it still needs to be fixed
    df.index = df['datetime']
    df.drop('datetime', axis='columns', inplace=True)
    start_date = str(start_date)
    end_date = str(end_date)
    
    if save:
        if save_path == '':
            if os.path.exists(f'data/{api}/{symbol}/'):
                DataFrame.to_csv(df, f'data/{api}/{symbol}/{symbol}_{start_date}_{end_date}.csv')
                return df, f'data/{api}/{symbol}/{symbol}_{start_date}_{end_date}.csv'
            else:
                os.makedirs(f'data/{api}/{symbol}/')
                DataFrame.to_csv(df, f'data/{api}/{symbol}/{symbol}_{start_date}_{end_date}.csv')
                return df, f'data/{api}/{symbol}/{symbol}_{start_date}_{end_date}.csv'
        else:
            if os.path.exists(f'data/{api}/'): 
                DataFrame.to_csv(df, save_path)
                return df, save_path
            else: 
                os.mkdir(f'data/{api}/')
                DataFrame.to_csv(df, save_path)
                return df, save_path
    else:
        return df
    # TODO: polygon.io API compatbility