from utils.api.queries import get_data
from api.creds import client_connect
from tda.client import Client
from data.dataset import StockDataGenerator
from models.tf.models import LongShortTermMemory
import unittest
import tracemalloc as tm
global verbose
import sys
import os


@unittest.skip('')
class TestBasics(unittest.TestCase):
    ''' Basic tests for various smaller components of the software '''
    def testGetData(self):
        tda_client = client_connect('TDA', 'private/creds.ini')
        self.assertEqual(type(tda_client), Client)
        if verbose > 0: print('\n[ TD Ameritrade API, Symbol: BLK ]')
        data = get_data(tda_client, symbol='BLK', 
                        period='TEN_YEAR', period_type='YEAR', 
                        frequency='DAILY', frequency_type='DAILY',
                        features= { 
                            'EMA': {}, '%B': {}, 'MIDPOINT': {}, 'CCI': {}, 'RSI': {}, 'VIX': {}, 'AROONOSC': {}
                        }, api='TDA', save=True, save_path='data/TDA/example.csv')
        if verbose > 0: print(data)
        tda_client.session.close()

class TestPipeline(unittest.TestCase):
    ''' Tests for data downloading, feature creation, and basic training pipelines '''
    @unittest.skip('minimize API calls')
    def testDatasetCompiling(self):
        data = StockDataGenerator(
            'BLK', 'TDA', 
            period='TEN_YEAR', period_type='YEAR', 
            frequency='DAILY', frequency_type='DAILY',
            features = {
                'EMA': {}, '%B': {}, 'RSI': {}
            }, save=True, verbose=verbose
        )
        data.client.session.close()
        if verbose > 0: print('\n', data.data)
        self.assertEqual(type(data), StockDataGenerator)
    
    # @unittest.skip('')
    def testModelCompileTrain(self):
        print('\n')
        data = StockDataGenerator(
            'BLK', 
            data_path='data/TDA/example.csv',
            verbose=verbose, target='close'
        )
        if verbose > 0: print(data.data)
        lstm = LongShortTermMemory()
        lstm.compile_model(data.X_train, verbose=verbose)
        lstm.train_model(data.X_train, data.y_train, epochs=25, verbose=verbose, plot_metrics=True)

if __name__ == '__main__':
    verbose = 1 if '-v' in sys.argv else 0
    unittest.main(verbosity=2)