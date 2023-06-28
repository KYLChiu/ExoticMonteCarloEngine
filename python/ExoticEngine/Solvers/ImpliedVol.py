import abc
from collections.abc import Callable
from enum import Enum
from inspect import isfunction
from typing import final

import numpy as np
from scipy.stats import norm

from ExoticEngine import MonteCarloPricer as Pricer


class PUT_CALL(Enum):
    PUT = "PUT"
    CALL = "CALL"


class FunctionObject:
    """
    Base class used for functions with 1 parameter
    If f is multidimensional, then need to define derived class
    """

    @abc.abstractmethod
    def __init__(self, function: Callable):
        """Child classes must define a different constructor"""
        assert isfunction(function)
        self._func = function

    @abc.abstractmethod
    def f(self, x: float) -> float:
        return self._func(x)

    def derivative(self, x: float):
        """not an abstract method - not all solvers require derivative"""
        raise Exception(
            "The derivative method needs to be overriden in a subclass if needed"
        )


@final
class BSModel(FunctionObject):
    """
    No dividend and repo rate
    """

    def __init__(
        self,
        put_call_flag: PUT_CALL,
        spot: float,
        strike: float,
        rate: float,
        maturity: float,
    ):
        self._S = spot
        self._K = strike
        self._r = rate
        self._T = maturity
        self._put_call = put_call_flag.value

    def f(self, sigma: float) -> float:
        if self._put_call == "PUT":
            return Pricer.BS_PUT(self._S, self._K, self._T, self._r, sigma)
        elif self._put_call == "CALL":
            return Pricer.BS_CALL(self._S, self._K, self._T, self._r, sigma)
        else:
            raise Exception(
                f"This is impossible: check put_call_flag: {self._put_call}"
            )

    def derivative(self, sigma: float) -> float:
        """returns: vega"""
        d1 = (np.log(self._S / self._K) + (self._r + sigma**2 / 2) * self._T) / (
            sigma * np.sqrt(self._T)
        )
        return self._S * np.sqrt(self._T) * norm.cdf(d1)


@final
class Polynomial(FunctionObject):
    def __init__(self, coefficients: tuple[float, int]):
        """
        Here, we use a tuple to store our coefficients,
        alternatively, we can also use *arg or **kwarg to support variable-length argument list
        see: https://www.geeksforgeeks.org/args-kwargs-python/
        """
        assert len(coefficients) > 0
        self._coefficients = coefficients

    def f(self, x: float) -> float:
        """returns: sum_{i=0} a_i x^i"""
        # try caching later: @functools.cache
        total = 0
        for i, a_i in enumerate(self._coefficients):
            total += a_i * (x ** float(i))
        return total

    def derivative(self, x: float) -> float:
        """returns: sum_{i=1} (a_i*i) x^(i-1)"""
        if x == 0:
            return self._coefficients[1] if len(self._coefficients) >= 2 else 0
        else:
            # try caching later: @functools.cache
            return sum(
                a_i * i * (x ** (i - 1))
                for i, a_i in enumerate(self._coefficients)
                if i > 0
            )


@final
class Exponential(FunctionObject):
    def __init__(self, factor: float, exponent: float, constant: float):
        self._A = factor
        self._k = exponent
        self._C = constant

    def f(self, x: float) -> float:
        return self._A * np.exp(self._k * x) + self._C

    def derivative(self, x: float) -> float:
        return self._A * self._k * np.exp(self._k * x)
