import abc

import numpy as np

from ExoticEngine import MonteCarloPricer as Pricer
from ExoticEngine.MarketDataObject.VolatilitySurfaces import \
    VolatilitySurface as VS


class LocalVolatilitySurface(abc.ABC):
    def __init__(self, implied_vol_surface: VS.VolatilitySurface):
        self._implied_vol_surface = implied_vol_surface

    @abc.abstractmethod
    def local_vol(self, K, T):
        pass


class BSLocalVolatilitySurface(LocalVolatilitySurface):
    """
    This local vol requires information from VolatilitySurfaceBlackScholes
    maybe this should just be child class of VolatilitySurfaceBlackScholes?
    """

    def __init__(self, implied_vol_surface: VS.VolatilitySurfaceBlackScholes):
        super.__init__(implied_vol_surface)
        self._r = implied_vol_surface.rate
        self._q = implied_vol_surface.repo_rate
        self._S0 = implied_vol_surface.spot
        self._Ks = implied_vol_surface.option_strikes
        self._Ts = implied_vol_surface.option_maturities
        self._forward = lambda T: self._S0 * np.exp(
            self._q.get_integral(0, T) - self._r.get_integral(0, T)
        )

    def __first_diff(self, f, x, dx):
        """
        returns df/dx
        finite diff is not a good idea
        maybe should input deriv function? make this configurable/extendable
        """
        return (f(x + dx) - f(x - dx)) / (2 * dx)

    def __second_diff(self, f, x, dx):
        """
        returns d^2f/dx^2
        finite diff is not a good idea
        maybe should input deriv function? make this configurable/extendable
        """
        return (f(x + dx) - 2 * f(x) + f(x - dx)) / dx**2

    def local_vol(self, K, T):
        """
        Instead using standard Dupire formula (based on option prem)
        It is more convienent/efficient to work with IV directly
        """
        log_moneyness = lambda k: np.log(k / self._forward(T))  # log_moneyness
        inverse_y = lambda Y: np.exp(Y) * self._forward(T)
        var_t = lambda t: self._implied_vol_surface.implied_vol(t, K) ** 2 * t
        var_y = (
            lambda Y: self._implied_vol_surface.implied_vol(T, inverse_y(Y)) ** 2 * T
        )
        dvdT = self.__first_diff(var_t, T, 1 / 365.0)
        dvdy = self.__first_diff(var_y, log_moneyness(K), 0.01)  # dy = dK/K = 0.01
        d2vdy2 = self.second_diff(var_y, log_moneyness(K), 0.01)
        v = var_t(T)  # implied BS total variance
        y = log_moneyness(K)
        denominator = (
            1
            - y / v * dvdy
            + 0.25 * (-0.25 - 1 / v + (y / v) ** 2) * dvdy**2
            + 0.5 * d2vdy2
        )
        return np.sqrt(dvdT / denominator)

    def local_vol_dupire(self, K, T):
        """
        Just for sanity checking: should give the same number as local_vol
        using Dupire formula explicitly
        """
        r = self._r.get_mean(0, T)
        q = self._q.get_mean(0, T)
        vol_t = lambda t: self._implied_vol_surface.implied_vol(t, K)
        vol_k = lambda k: self._implied_vol_surface.implied_vol(T, k)
        C_t = lambda t: Pricer.BS_CALL(self._S0, K, t, r, vol_t(t))
        C_k = lambda k: Pricer.BS_CALL(self._S0, K, T, r, vol_k(k))
        dCdT = self.__first_diff(C_t, T, 1 / 365.0)
        dCdK = self.__first_diff(C_k, K, 0.01 * K)
        d2CdK2 = self.second_diff(C_k, K, 0.01 * K)
        numerator = dCdT + q * C_t(T) + (r - q) * K * dCdK
        denominator = 0.5 * K**2 * d2CdK2
        return np.sqrt(numerator / denominator)
