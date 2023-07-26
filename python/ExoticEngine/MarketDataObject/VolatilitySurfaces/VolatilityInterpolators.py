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
        self._XX, self._YY = np.meshgrid(self._X, self._Y)

    @abc.abstractmethod
    def interpolation_2d(self, x, y):
        pass


class VolatilityInterpolationBilinear(VolatilityInterpolation):
    def __init__(self, x_axis, y_axis, z_values):
        super().__init__(x_axis, y_axis, z_values)

    def interpolation_2d(self, x, y):
        tck = interpolate.bisplrep(self._XX, self._YY, self._Z, s=0)
        return interpolate.bisplev(x, y, tck)
