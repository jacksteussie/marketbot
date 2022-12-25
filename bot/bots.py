import asyncio
from strategies import Strategy
from backtesting import datafeed
from multiprocessing import Pool, cpu_count, Manager, Process, Queue

class Bot:
    def __init__(self, strategy, auto_update_model: bool):
        self.strategy = strategy
        self.auto_update_model = auto_update_model
        self.active = False
    
    @property
    def status(self):
        # TODO:
        return self._active
    
    @status.setter
    def toggle(self):
        self._active = not self._active
        # TODO:

    

    def toggle_model_auto_update(self):
        if self.auto_update_model is True:
            print('Toggling automated model training and updating OFF')
            self.auto_update_model = False
            # TODO:
        else:
            print('Toggling automated model training and updating ON')
            self.auto_update_model = True
            #TODO:

class DayTradingBot(Bot):
    def __init__(self, strategy, auto_update_model: bool):
        super().__init__(strategy, auto_update_model)

class SwingTradingBot(Bot):
    def __init__(self, strategy, auto_update_model: bool):
        super().__init__(strategy, auto_update_model)

class PositionTradingBot(Bot):
    def __init__(self, strategy, auto_update_model: bool):
        super().__init__(strategy, auto_update_model)

class ClusterWatch(Bot):
    def __init__(self, strategy, auto_update_model: bool):
        super().__init__(strategy, auto_update_model)
