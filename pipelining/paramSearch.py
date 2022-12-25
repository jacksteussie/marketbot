from models.tf.models import LongShortTermMemoryR
from joblib import Memory
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import absl.logging
from time import sleep, time
import numpy as np
from sklearn.manifold import Isomap
import xgboost
import tensorflow as tf
import scipy
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
from omegaconf import OmegaConf
import hydra

class Pipeline():
    def __init__(self, models, data, target, features, **kwargs):
        self.models = models
        self.data = data
        self.target = target
        self.features = features
        self.kwargs = kwargs
        self.verbose = kwargs.get('verbose', 0)
        self.memory = Memory(location='cache', verbose=0)
        self.logger = absl.logging.get_absl_logger()
        self.logger.set_verbosity(self.verbose)
        self.logger.info('Initialized Pipeline')
        self.logger.info('Models: {}'.format(self.models))

# 1. Define an objective function to be maximized.
    def objective(trial):
        model = trial.suggest_categorical('model', ['lstm', 'xgbr', 'xgbrf'])
        features = trial.suggest_categorical('features', features)

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)