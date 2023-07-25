import abc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


class VolatilityInterpolation(abc.ABC):
    """
    agnostic to IV or option premium data
    """

    def __init__(self, x_axis, y_axis, z_values):
        self._X = x_axis
        self._Y = y_axis
        self._Z = z_values

    @abc.abstractmethod
    def interpolation_2d(self, x, y, surface):
        pass


class VolatilityInterpolationBilinear(VolatilityInterpolation):
    def __init__(self, x_axis, y_axis, z_values):
        super.__init__(x_axis, y_axis, z_values)

    def interpolation_2d(self, x, y):
        tck = interpolate.bisplrep(self._X, self._Y, self._Z, s=0)
        return interpolate.bisplev(x, x, tck)
