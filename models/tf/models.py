from math import sqrt, floor
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense, Softmax
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from matplotlib import pyplot as plt
import numpy as np
from queue import Queue
import sys
import shap

class LongShortTermMemoryR(tf.keras.Model):
    '''
        LSTM for regression/forecasting tasks
    '''
    def __init__(self, loss='mse', opt='Adam', target='close', method='regression', enable_checkpoints=False):
        super().__init__()
        self.target = target
        self.model = None
        self.loss = loss
        self.optimizer = opt
        self.enable_checkpoints = enable_checkpoints
        self.method = method


    @property
    def metrics(self):
        metrics = [  
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
        return metrics


    @property
    def callbacks(self):
        callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1), 
        ]
        # if self.enable_checkpoints:
            # callbacks.append(
            #     tf.keras.callbacks.ModelCheckpoint(
            #         filepath='models/tf/checkpoints/model.{epoch:02d}-{loss:.4f}.h5', 
            #         verbose=self.callback_verbose
            #     )
            # )
        return callbacks
     

    def info(self, summary=True): 
        self.model.summary()
        return 


    def compile_model(self, X_train, prediction_range=200, verbose=0, name=None) -> tf.keras.Model:
        model = Sequential(name=name)

        for 
        
        # 1st LSTM layer
        # * units = add prediction_range number of neurons (the dimensionality of the output space)
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        # * input_shape -> Shape of the training dataset
        model.add(LSTM(units=prediction_range, return_sequences=True, input_shape=(X_train.shape[1], 1)))

        # 1st Dropout layer: 20% of the layers will be dropped
        model.add(Dropout(0.2))
        
        # 2nd LSTM layer
        # * units = add prediction_range / 2 number of neurons (the dimensionality of the output space)
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        model.add(LSTM(units=prediction_range // 2, return_sequences=True))

        # 2nd Dropout layer: 20% of the layers will be dropped
        model.add(Dropout(0.2))

        # 3rd LSTM layer
        # * units = add prediction_range / 2 number of neurons (the dimensionality of the output space)
        # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
        model.add(Bidirectional(LSTM(units=prediction_range // 2, return_sequences=True)))

        # 3rd Dropout layer: 50% of the layers will be dropped
        model.add(Dropout(0.5))

        # 4th LSTM layer
        # * units = add 50 neurons is the dimensionality of the output space
        model.add(Bidirectional(tf.keras.layers.LSTM(units=prediction_range // 2)))

        model.add(Dropout(0.5))

        # Dense layer that specifies an output of one unit
        model.add(Dense(units=1))

        
        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if verbose > 0: model.summary()
        return self.model


    def train_model(self, X_train, y_train, X_val=None, y_val=None, plot_metrics=False, epochs=25, verbose=0):
        self.callback_verbose = verbose 
        if self.model is None:
            raise TypeError('Model has not been created yet! (received None type as input to trainer)')
        if X_val is None or y_val is None:
            self.train_history = self.model.fit(
                X_train, y_train, epochs=epochs,
                verbose=verbose, callbacks=self.callbacks, 
                use_multiprocessing=True)
        else:
            self.train_history = self.model.fit(
                X_train, y_train, epochs=epochs,
                validation_data=(X_val, y_val),
                verbose=verbose, callbacks=self.callbacks, 
                use_multiprocessing=True)
        
        if plot_metrics == True:
            self.plot_train_results()
            self.show_train_pred(X_train, y_train)
            self.get_shap(X_train, X_val)


    def test_model(self, X_test, y_test, verbose=0):
        self.test_metrics = self.model.evaluate(X_test, y_test, verbose=verbose, callbacks=self.callbacks)


    def show_train_pred(self, X_train, y_train):
        y_pred = self.model.predict(X_train)
        fig = plt.figure(2)
        plt.plot(range(len(y_train)), y_train, label='y_truth')
        plt.plot(range(len(y_pred)), y_pred, label='y_pred')
        plt.legend()
        plt.show()


    def plot_train_results(self):
        '''
            Algorithm to create a square of plots (with potentially the last row missing)
            one or two graphs. The plots contain the progression of the training metrics 
            used for the model.
            Algorithm: 
                (Setup) Let m, n be the value of a single one of our square's dimensions,
                        the total number of plots, respectively.
                (1) Need to find what consecutive squares n lies between (i.e. m^2 <= n <= (m + 1)^2).
                    Find m (i.e. floor(sqrt(n))).
                (2) Let k be the surplus of blocks on the last row.
                    Calculate the surplus (i.e. k = m^2 - n)
                (3) If 0 <= k < m, form (m + 1, m) square with the last row containing k plots. 
                    Else if m <= k < 2m + 1, form (m + 1, m + 1) with the last row containing k plots.
                (End)

        '''
        epochs = [ i for i in range(len(self.train_history.history['loss'])) ]
        metrics = [ metric.name for metric in self.metrics ]
        n = len(self.metrics) + 1   # number of metrics plus loss
        m = floor(sqrt(n))
        k = abs(m ** 2 - n)
        metrics_queue = Queue(n) # FIFO
        metrics_queue.put('loss')
        for metric in metrics: metrics_queue.put(metric)
        if n == 1:
            fig = plt.plot(epochs, self.train_history.history['loss'])  
            plt.savefig('models/logs/train_graphs.png')
            print(fig)
            return fig
        elif k == 0:
            fig, ax = plt.subplots(m, m)
            for r in range(m):
                for c in range(m):
                    if metrics_queue.empty():
                        break
                    else:
                        metric = metrics_queue.get(block=True)
                        ax[r][c].set_title(f'Training {metric}')
                        ax[r][c].plot(epochs, self.train_history.history[metric])
        if 0 < k < m: 
            fig, ax = plt.subplots(m + 1, m)
            fig.tight_layout()

            for r in range(m + 1):
                if r == m:
                    for c in range(k):
                        if metrics_queue.empty():
                            break
                        else:
                            metric = metrics_queue.get(block=True)
                            ax[r][c].set_title(f'Training {metric}')
                            ax[r][c].plot(epochs, self.train_history.history[metric])
                else:
                    for c in range(m):
                        if metrics_queue.empty():
                            break
                        else:
                            metric = metrics_queue.get(block=True)
                            ax[r][c].set_title(f'Training {metric}')
                            ax[r][c].plot(epochs, self.train_history.history[metric])
        elif m <= k < 2 * m + 1:
            fig, ax = plt.subplots(m + 1, m + 1)
            fig.tight_layout()

            for r in range(m + 1):
                if r == m + 1:
                    for c in range(k):
                        if metrics_queue.empty():
                            break
                        else:
                            metric = metrics_queue.get(block=True)
                            ax[r][c].set_title(f'Training {metric}')
                            ax[r][c].plot(epochs, self.train_history.history[metric])
                else:
                    for c in range(m + 1):
                        if metrics_queue.empty():
                            break
                        else:
                            metric = metrics_queue.get(block=True)
                            ax[r][c].set_title(f'Training {metric}')
                            ax[r][c].plot(epochs, self.train_history.history[metric])

        fig.savefig(f'models/logs/train_graphs_{self.model.name}.png')
        return fig


    def get_shap(self, X_train, X_test):
        explainer = shap.DeepExplainer(self.model, X_train[:])
        shap_values = explainer.shap_values(X_test[:])
        shap.summary_plot(shap_values, X_train)
        # TODO: