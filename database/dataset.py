from dataclasses import dataclass
from typing import Any, Union
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from utils.api.connections import client_connect
from utils.api.queries import get_data
import os 

class StockDataGenerator():

    def __init__(self, symbol=None, api='', data_path='', features={}, 
                 target='close', num_steps=30, test_ratio=0.15, normalized=True,
                 close_price_only=True, verbose=0, period: Any=None, 
                 period_type: Any=None, frequency: Any=None, frequency_type: Any=None, 
                 save=False, lookback=0, up_down=False, method='regression', **kwargs):
        self.symbol = symbol
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized
        self.client = None
        self.features = features
        self.api = api
        self.period = period
        self.period_type = period_type
        self.frequency = frequency
        self.frequency_type = frequency_type
        self.save = save
        self.verbose = verbose
        self.lookback = lookback
        self.method = method


        if data_path == '':
            if self.api == 'TDA':
                self.client = client_connect(self.api, 'private/creds.ini')
            elif self.api == 'POLY':
                self.client = client_connect(self.api, 'private/creds.ini')

            if self.api == 'TDA':
                if self.save:
                    self.data, self.data_path = get_data(
                        client=self.client, api=self.api, features=self.features, symbol=self.symbol,
                        save=self.save, period=self.period, period_type=self.period_type, frequency=self.frequency, 
                        frequency_type=self.frequency_type 
                    )
                else:
                    self.data = get_data(
                        client=self.client, api=self.api, features=self.features, symbol=self.symbol,
                        save=self.save, period=self.period, period_type=self.period_type, frequency=self.frequency, 
                        frequency_type=self.frequency_type 
                    )
            elif self.api == 'POLY':
                self.data = get_data(
                    client=self.client, api=self.api, 
                    features=self.features, symbol=self.symbol,
                    save=self.save, **kwargs
                )
        else:
            self.data = pd.read_csv(data_path, index_col='datetime')
            self.symbol = os.path.basename(data_path)

        self.num_features = len(self.data.columns) - 1 
        self.target = target
        self._process_dataset()
        self._train_test_split()


    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.symbol, len(self.train_X), len(self.test_y)
        )


    def get_num_features(self): return self.num_features


    def _process_dataset(self, normalization=True, pc=False):
        if self.method == 'classification':
            self.data['close_class'] = self.data['close'].pct_change()
            self.data['close_class'] = self.data['close'].apply(lambda x: -1 if x < 0 else 1)
            self.target = 'close_class'
        if self.verbose >= 1: print('Normalizing data...')
        if pc:
            self.data['percentChangeOC'] = self.data['close'] / self.data['open'] - 1
            if self.verbose >= 1 and self.lookback > 0: print('Generating lookback features...')        
            for i in range(1, self.lookback + 1):
                self.data[f'percentChange-{i}'] = self.data['percentChangeOC'].shift(-self.lookback)
        if self.verbose >= 1: print('Scaling data...')
        self.data = pd.DataFrame(scale(X=self.data), index=self.data.index, columns=self.data.columns)
    

    def _train_test_split(self, test_percent=0.15):
        self.X = self.data.loc[:, self.data.columns != self.target]
        self.y = self.data.loc[:, self.data.columns == self.target]
        test_size = int(len(self.data) * test_percent)
        X_train = self.X.head(len(self.X) - test_size)
        y_train = self.y.head(len(self.y) - test_size)
        X_test = self.X.head(test_size)
        y_test = self.y.head(test_size)
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.X_test = X_test.to_numpy()
        self.y_test = y_test.to_numpy()


    def _generate_window():
        # TODO:
        pass


    def plot_features(self, features: list):
        for feat in features:
            feat = self.data
        figures = []
        for i in self.data.columns.values:
            pass
    
    def fit():
        pass

