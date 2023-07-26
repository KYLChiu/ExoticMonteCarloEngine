import abc
from collections.abc import Callable
from inspect import isfunction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy import interpolate

from ExoticEngine.MarketDataObject import Parameter as P
from ExoticEngine.MarketDataObject.VolatilitySurfaces import \
    VolatilityInterpolators as VI


class VolatilitySurface(abc.ABC):
    def __init__(self, option_quotes: pd.DataFrame, interpolation_style):
        self._quotes = option_quotes
        self._strikes = option_quotes.columns.astype(float)
        self._maturities = option_quotes.index.astype(float)
        self.__K, self.__T = np.meshgrid(self._strikes, self._maturities)
        if interpolation_style == "bilinear":
            self._interpolator = VI.VolatilityInterpolationBilinear(
                self._strikes, self._maturities, self._quotes
            )
        else:
            raise Exception(f"Only bilinear implemented: {interpolation_style}")

    def plot_surface(self, cmap=cm.coolwarm):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            self.__T, self.__K, self._quotes, cmap=cmap, linewidth=0, antialiased=False
        )

    def plot_contour(self):
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(self.__T, self.__K, self._quotes)
        fig.colorbar(cp)  # Add a colorbar to a plot
        ax.set_title("Vol surface")
        plt.show()

    @abc.abstractmethod
    def implied_vol(self, T, K):
        pass

    @property
    def option_strikes(self):
        return self._strikes

    @property
    def option_maturities(self):
        return self._maturities


class VolatilitySurfaceBlackScholes(VolatilitySurface):
    def __init__(
        self,
        option_quotes: pd.DataFrame,
        rate: P.Parameter,
        repo_rate: P.Parameter,
        spot: float,
        interpolation_style: str,
    ):
        """Assumes taking in an implied vol surface"""
        self._r = rate
        self._q = repo_rate
        self._S0 = spot
        super().__init__(option_quotes, interpolation_style)

    def implied_vol(self, T, K):
        """
        if T,K in R, then return sigma(T,K) in R
        elif T,K are vectors, then return implied vol surface
        """
        return self._interpolator.interpolation_2d(T, K)

    @property
    def rate(self):
        return self._r

    @property
    def repo_rate(self):
        return self._q

    @property
    def spot(self):
        return self._S0
