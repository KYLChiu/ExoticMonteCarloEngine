import numpy as np
import pathos.multiprocessing as mp

from ExoticEngine.Payoff import Options as O
from ExoticEngine.MonteCarloEngine import SimulationModel as Sim
from python.sandbox import pricer_helper as helper
from python.ExoticEngine.MonteCarloEngine import VanillaPricer

def zero_rate_zero_vol_impl(num_cores: int = 1):
    spots = [90, 99, 100, 110]
    pay_off = O.PayOffPut(strike=100)
    option = O.VanillaOption(pay_off, expiry=2)
    param_dict = helper.build_constant_market_param(0, 0, 0, 0)
    RNGenerator = helper.build_RNG("PSEUDO_RANDOM")
    prices = [10, 1, 0, 0]
    for i, S in enumerate(spots):
        EqModel = Sim.BSModel(spot=S,
                              r=param_dict["Rate"],
                              repo_rate=param_dict["Repo"],
                              div_yield=param_dict["Div"],
                              vol=param_dict["Vol"],
                              RNG=RNGenerator)
        collector = helper.build_collector("PATH_COUNT", 10)
        pricer = VanillaPricer.VanillaMCPricer(option,
                                           param_dict["Rate"],
                                           EqModel,
                                           collector, num_cores)
        pricer.price()
        assert pricer.stats.get_mean() == prices[i]
        assert pricer.stats.get_std_err() == 0.
        assert pricer.stats.get_path_count() == 10


def test_vanilla_put_pricer_zero_rate_zero_vol_sp():
    zero_rate_zero_vol_impl(1)

def test_vanilla_put_pricer_zero_rate_zero_vol_mp():
    zero_rate_zero_vol_impl(2)


def zero_vol_impl(num_cores: int = 1):
    maturity = 5.
    K = 100
    spots = [90, 100, 101, 110]
    pay_off = O.PayOffCall(strike=K)
    option = O.VanillaOption(pay_off, expiry=maturity)
    param_dict = helper.build_constant_market_param(rate=0.05, vol=0., repo=0.02, div=0.01)
    RNGenerator = helper.build_RNG("PSEUDO_RANDOM")
    num_path = 10
    for S in spots:
        EqModel = Sim.BSModel(spot=S,
                            r=param_dict["Rate"],
                            repo_rate=param_dict["Repo"],
                            div_yield=param_dict["Div"],
                            vol=param_dict["Vol"],
                            RNG=RNGenerator)
        collector = helper.build_collector("PATH_COUNT", num_path)
        pricer = VanillaPricer.VanillaMCPricer(option,
                                           param_dict["Rate"],
                                           EqModel,
                                           collector, num_cores)
        pricer.price()
        assert pricer.stats.get_std_err() == 0.
        assert pricer.stats.get_path_count() == num_path

def test_vanilla_call_pricer_zero_vol_sp():
    zero_vol_impl(1)


def test_vanilla_call_pricer_zero_vol_mp():
    zero_vol_impl(2)
   
