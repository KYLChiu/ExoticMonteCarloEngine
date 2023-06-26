import abc
from enum import Enum
from typing import final

import numpy as np
from scipy.stats import norm

from ExoticEngine import MonteCarloPricer as Pricer
from ExoticEngine.MarketDataObject import Parameter as P


class PUT_CALL(Enum):
    PUT = "PUT"
    CALL = "CALL"


class InvertFunction:
    """
    Base class used for functions with 1 parameter
    If f is multidimensional, then need to define derived class
    """

    def __init__(self, function):
        self._func = function

    @abc.abstractmethod
    def f(self, x: float):
        return self._func(x)

    def derivative(self, x: float):
        """not an abstract method - not all solvers require derivative"""
        return None


@final
class BSModel(InvertFunction):
    """
    No dividend and repo rate
    """

    def __init__(
        self,
        put_call_flag: str,
        spot: float,
        strike: float,
        rate: float,
        maturity: float,
    ):
        self._S = spot
        self._K = strike
        self._r = rate
        self._T = maturity
        self._put_call = PUT_CALL(put_call_flag)

    def f(self, sigma: float):
        if self._put_call.value == "PUT":
            return Pricer.BS_PUT(self._S, self._K, self._T, self._r, sigma)
        elif self._put_call.value == "CALL":
            return Pricer.BS_CALL(self._S, self._K, self._T, self._r, sigma)
        else:
            raise Exception(
                f"This is impossible: check put_call_flag: {self._put_call.value}"
            )

    def derivative(self, sigma: float):
        """returns: vega"""
        d1 = (np.log(self._S / self._K) + (self._r + sigma**2 / 2) * self._T) / (
            sigma * np.sqrt(self._T)
        )
        return self._S * np.sqrt(self._T) * norm.cdf(d1)


@final
class Polynomial(InvertFunction):
    def __init__(self, coefficients: list[float]):
        assert len(coefficients) > 0
        self._coefficients = coefficients

    def f(self, x: float) -> float:
        """returns: sum_{i=0} a_i x^i"""
        total = 0
        for i, a_i in enumerate(self._coefficients):
            total += a_i * (x ** float(i))
        return total

    def derivative(self, x: float):
        """returns: sum_{i=1} (a_i*i) x^(i-1)"""
        if x == 0:
            return self._coefficients[1]
        else:
            total = 0
            for i, a_i in enumerate(self._coefficients):
                total += (a_i * float(i)) * (x ** (float(i) - 1))
            return total


@final
class Exponential(InvertFunction):
    def __init__(self, factor: float, exponent: float, constant: float):
        self._A = factor
        self._k = exponent
        self._C = constant

    def f(self, x):
        return self._A * np.exp(self._k * x) + self._C

    def derivative(self, x: float):
        return self._A * self._k * np.exp(self._k * x)
