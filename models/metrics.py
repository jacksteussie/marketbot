import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric

class SharpeRatio(Metric):
    '''
        The Sharpe ratio compares the return of an investment with its risk. 
        It's a mathematical expression of the insight that excess returns over 
        a period of time may signify more volatility and risk, rather than 
        investing skill. The Sharpe ratio is calculated by subtracting the risk-free
        rate from the return of the investment and dividing the result by the standard
        deviation of the investment. The higher the Sharpe ratio, the better the investment.
    '''

    def __init__(self, name='SharpeRatio', **kwargs):
        super(SharpeRatio, self).__init__(name=name, **kwargs)
        self.add_weight(name='portfolioReturn', initializer='zeros')
        self.add_weight(name='riskFreeRate', initializer='zeros')
        self.add_weight(name='portfolioVolatility', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        return super().update_state(*args, **kwargs)
    
    def result(self):
        return (self.portfolioReturn - self.riskFreeRate) / self.portfolioVolatility

class CapitalAssetPricingModel(Metric):
    '''
        Capital Asset Pricing Model (CAPM) is a model that describes the relationship
        between expected return and risk. It is used to determine a theoretically
        appropriate required rate of return of an asset, to make decisions about
        adding assets to a well-diversified portfolio. The model is based on the
        idea that investors require a certain rate of return for bearing the risk
        of investing in a security. The required rate of return is the minimum
        rate of return that an investor requires to compensate for the risk of
        investing in a security. The required rate of return is the risk-free rate
        plus a risk premium. The risk premium is the additional return an investor
        requires to compensate for the additional risk of investing in a security
        rather than a risk-free asset. The risk premium is the difference between
        the expected return of the security and the risk-free rate. The CAPM is
        based on the assumption that investors are rational and that they will
        demand a risk premium for bearing the risk of investing in a security.
    '''

    def __init__(self, name='CapitalAssetPricingModel', **kwargs):
        super(CapitalAssetPricingModel, self).__init__(name=name, **kwargs)
        self.add_weight(name='expectedReturnI', initializer='zeros')
        self.add_weight(name='expectedReturnM', initializer='zeros')
        self.add_weight(name='riskFreeRate', initializer='zeros')
        self.add_weight(name='beta', initializer='zeros')
        self.add_weight(name='riskPremium', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        return super().update_state(*args, **kwargs)

    def result(self):
        return self.riskFreeRate + self.beta * (self.expectedReturnM - self.riskFreeRate)
    
class CalmarRatio(Metric):
    '''
        The Calmar ratio is a measure of risk-adjusted performance for investment managers.
        It is a measure of the ratio of the average annual rate of return of an investment
        to its maximum drawdown. The Calmar ratio is a measure of the risk-adjusted return
        of an investment. The higher the Calmar ratio, the better the investment. The Calmar
        ratio is calculated by dividing the average annual rate of return of an investment
        by its maximum drawdown. The maximum drawdown is the largest decline in the value of
        an investment from its peak value. The maximum drawdown is the largest decline in the
        value of an investment from its peak value. The maximum drawdown is the largest decline
        in the value of an investment from its peak value. The maximum drawdown is the largest
        decline in the value of an investment from its peak value. The maximum drawdown is the
        largest decline in the value of an investment from its peak value. The maximum drawdown
        is the largest decline in the value of an investment from its peak value. The maximum
        drawdown is the largest decline in the value of an investment from its peak value. The
        maximum drawdown is the largest decline in the value of an investment from its peak value.
    '''

    def __init__(self, name='CalmarRatio', **kwargs):
        super(CalmarRatio, self).__init__(name=name, **kwargs)
        self.add_weight(name='averageAnnualReturn', initializer='zeros')
        self.add_weight(name='maximumDrawdown', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None, *args, **kwargs):
        return super().update_state(*args, **kwargs)

    def result(self):
        return self.averageAnnualReturn / self.maximumDrawdown
    
