import abc
import numpy as np
from typing import final
from ExoticEngine.Statistics import RandomNumberGenerators as RNG

from ExoticEngine.MarketDataObject import Parameter as P

class Model(abc.ABC):
    @abc.abstractmethod
    def sde(self) -> float:
        pass

@final
class BSModel(Model):
    def __init__(self, spot: float, r: float, vol: float, RNG: type[RNG.RandomBase]):
        """Smile NOT included yet"""
        self._spot = spot  # constant
        self._r = r  # constant
        self._vol = vol  # constant
        self._RNG = RNG

    def sde(self, dt) -> float:
        """
        returns: dS = ...
        GBM has exact solution - dS = S(t) - S(0) = ...
        single path only...would it be slow?
        """
        ito = 0.5 * self._vol ** 2
        drift = (self._r - ito) * dt
        diffusion = self._vol * np.sqrt(dt) * self._RNG.get_gaussian()
        return self._spot * (np.exp(drift + diffusion) - 1)
