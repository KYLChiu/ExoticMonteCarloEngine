import scipy as sci
import numpy as np
from inspect import isfunction
from collections.abc import Callable


class Parameter:
    def __init__(self, param: Callable):
        """
        Takes in a deterministic function
        not the best design for now...
        should really be taking in the "Model" object
        """
        assert isfunction(param)
        self._param = param

    def get_mean(self, t1: float, t2: float) -> float:
        """returns: int_t1^t2 param(t) dt / (t2-t1)"""
        assert t2 >= t1
        assert t1 >= 0
        return sci.integrate.quad(self._param, t1, t2)[0] / (t2 - t1)

    def get_root_mean_square(self, t1: float, t2: float) -> float:
        """returns: int_t1^t2 param(t)^2 dt / (t2-t1)"""
        assert t2 >= t1
        assert t1 >= 0
        integrand = lambda t: self._param(t) ** 2
        return np.sqrt(sci.integrate.quad(integrand, t1, t2)[0] / (t2 - t1))
