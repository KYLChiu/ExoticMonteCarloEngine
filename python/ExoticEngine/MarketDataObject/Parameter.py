import scipy as sci
import numpy as np

class Parameter:
    def __init__(self, param):
        """
        Takes in a function
        not the best design for now...
        should really be taking in the "Model" object (with sde method)
        """
        self._param = param

    def get_mean(self, t1, t2):
        """returns: int_t1^t2 param(t) dt / (t2-t1)"""
        assert t2 >= t1
        integrand = lambda t: self._param(t)
        return sci.integrate.quad(self._param, t1, t2)[0] / (t2-t1)

    def get_root_mean_square(self, t1, t2):
        """returns: int_t1^t2 param(t)^2 dt / (t2-t1)"""
        assert t2 >= t1
        integrand = lambda t: self._param(t)**2
        return np.sqrt( sci.integrate.quad(integrand, t1, t2)[0] / (t2-t1) )
