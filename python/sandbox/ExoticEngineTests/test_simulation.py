from ExoticEngine.MonteCarloEngine import SimulationModel as Sim
from sandbox.ExoticEngineTests import pricer_helper as helper
import numpy as np


def test_BS_simulation_zero_rate_zero_vol():
    param_dict = helper.build_constant_market_param(rate=0., vol=0., repo=0.0, div=0.0)
    RNGenerator = helper.build_RNG("PSEUDO_RANDOM")
    spot = 100
    EqModel = Sim.BSModel(spot=spot,
                          r=param_dict["Rate"],
                          repo_rate=param_dict["Repo"],
                          div_yield=param_dict["Div"],
                          vol=param_dict["Vol"],
                          RNG=RNGenerator)
    assert EqModel.sde(0, 5) == 0


def test_BS_simulation_zero_vol(tolerance=1e-8):
    param_dict = helper.build_constant_market_param(rate=.05, vol=0., repo=0.03, div=0.)
    RNGenerator = helper.build_RNG("PSEUDO_RANDOM")
    maturity = 5.
    spot = 100
    EqModel = Sim.BSModel(spot=spot,
                          r=param_dict["Rate"],
                          repo_rate=param_dict["Repo"],
                          div_yield=param_dict["Div"],
                          vol=param_dict["Vol"],
                          RNG=RNGenerator)
    discount_factor_r = np.exp(-param_dict["Rate"].get_mean(0, maturity) * maturity)
    discount_factor_q = np.exp(-param_dict["Repo"].get_mean(0, maturity) * maturity)
    discount_factor_d = np.exp(-param_dict["Div"].get_mean(0, maturity) * maturity)
    effective_DF = discount_factor_r / discount_factor_q / discount_factor_d
    assert abs(EqModel.sde(0, maturity) - spot * (1 / effective_DF - 1)) < tolerance
