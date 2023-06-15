import abc
from abc import ABC
class PayOff(ABC):
    @abc.abstractmethod
    def payoff(self):
        pass

class PayOffCall(PayOff):
    def __init__(self, strike):
        self.strike_ = strike

    def payoff(self, spot):
        return max(spot-self.strike_, 0)

class PayOffPut(PayOff):
    def __init__(self, strike):
        self.strike_ = strike

    def payoff(self, spot):
        return max(self.strike_-spot, 0)

class VanillaOption:
    def __init__(self, PayOff: PayOff, expiry: float):
        self.PayOff_ = PayOff
        self.expiry_ = expiry

    def get_expiry(self):
        return self.expiry_

    def get_payoff(self, spot):
        return self.PayOff_.payoff(spot)







