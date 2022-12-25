from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=XGBRegressor()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        prob = self.estimator.predict_proba(X)
        return prob


    def score(self, X, y):
        score = self.estimator.score(X, y)
        print(score)
        return score