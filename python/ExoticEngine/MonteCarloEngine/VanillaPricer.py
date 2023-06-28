import numpy as np
from typing import final

from ExoticEngine.Payoff import Options as O
from ExoticEngine.Statistics import Statistics as Stats
from ExoticEngine.MonteCarloEngine import SimulationModel as Sim
from ExoticEngine.MonteCarloEngine import MonteCarloPricer
from ExoticEngine.MarketDataObject import Parameter as P
from scipy.stats import norm

@final
class VanillaMCPricer(MonteCarloPricer.MCPricer):

    def __init__(self, option: O.VanillaOption,
                 rate: P.Parameter,
                 sim_model: Sim.Model,
                 result_collector: Stats.GetStatistics, num_cores = 1):
        super().__init__(result_collector, num_cores)
        self._option = option
        self._rate = rate
        self._model = sim_model

    def _generate_path(self):
        expiry = self._option.get_expiry()
        discount_factor = np.exp(-self._rate.get_mean(0., expiry) * expiry)
        spot_t0 = self._model.get_day0()
        return discount_factor * self._option.get_payoff(self._model.sde(0, expiry) + spot_t0)


def BS_CALL(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BS_PUT(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

