from ExoticEngine.MarketDataObject import Parameter as P
from ExoticEngine import MonteCarloPricer as Pricer
from enum import Enum
import abc
from typing import final


class PUT_CALL(Enum):
    PUT = "PUT"
    CALL = "CALL"


class InverseFunctionObject(abc.ABC):
    abc.abstractmethod
    def f(self, x: float):
        pass

@final
class BSModel(InverseFunctionObject):
    """
    Function Object
    """
    def __init__(self,
                 put_call_flag: str,
                 spot: float,
                 strike: float,
                 maturity: float,
                 rate: P.Parameter,
                 repo: P.Parameter = P.Parameter(lambda t: 0),
                 dividend: P.Parameter = P.Parameter(lambda t: 0)):
        self._S = spot
        self._K = strike
        self._rate = rate
        self._q = repo
        self._d = dividend
        self.T = maturity
        self._put_call = PUT_CALL(put_call_flag)

    def f(self, x: float):
        if self._put_call.value == "PUT":
            return Pricer.BS_PUT(self._S, self._K, self._T, self._r, x)
        elif self._put_call.value == "CALL":
            return Pricer.BS_CALL(self._S, self._K, self._T, self._r, x)
        else:
            raise Exception(f"This is impossible: check put_call_flag: {self._put_call.value}")

