import abc
from typing import final

class PayOff(abc.ABC):
    @abc.abstractmethod
    def payoff(self: float) -> float:
        pass

@final
class PayOffCall(PayOff):
    def __init__(self, strike):
        self._strike = strike

    def payoff(self, spot: float) -> float:
        return max(spot-self._strike, 0)

@final
class PayOffPut(PayOff):
    def __init__(self, strike):
        self._strike = strike

    def payoff(self, spot: float) -> float:
        return max(self._strike-spot, 0)

@final
class PayOffDoubleDigital(PayOff):
    def __init__(self, upper, lower):
        self._upper = upper
        self._lower = lower
        assert self._lower < self._upper

    def payoff(self, spot: float) -> float:
        if spot > self._upper or spot < self._lower:
            return 0
        else:
            return 1

class VanillaOption:
    def __init__(self, pay_off: type[PayOff], expiry: float):
        self._pay_off = pay_off
        self._expiry = expiry

    def get_expiry(self) -> float:
        return self._expiry

    def get_payoff(self, spot: float) -> float:
        return self._pay_off.payoff(spot)






