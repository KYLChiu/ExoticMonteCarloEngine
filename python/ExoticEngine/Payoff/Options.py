import abc

class PayOff(abc.ABC):
    @abc.abstractmethod
    def payoff(self):
        pass

class PayOffCall(PayOff):
    def __init__(self, strike):
        self._strike = strike

    def payoff(self, spot):
        return max(spot-self._strike, 0)

class PayOffPut(PayOff):
    def __init__(self, strike):
        self._strike = strike

    def payoff(self, spot):
        return max(self._strike-spot, 0)

class PayOffDoubleDigital(PayOff):
    def __init__(self, upper, lower):
        self._upper = upper
        self._lower = lower
        assert self._lower < self._upper

    def payoff(self, spot):
        if spot > self._upper or spot < self._lower:
            return 0
        else:
            return 1

class VanillaOption:
    def __init__(self, PayOff: PayOff, expiry: float):
        self._PayOff = PayOff
        self._expiry = expiry

    def get_expiry(self):
        return self._expiry

    def get_payoff(self, spot):
        return self._PayOff.payoff(spot)






