import numpy as np

from ExoticEngine.Payoff import Options as O
from ExoticEngine.MonteCarloEngine import SimulationModel as Sim
from sandbox.ExoticEngineTests import pricer_helper as helper
from ExoticEngine import MonteCarloPricer as Pricer


def test_vanilla_put_pricer_zero_rate_zero_vol():
    spots = [90, 99, 100, 110]
    K, T = 100, 2.1
    pay_off = O.PayOffPut(strike=K)
    option = O.VanillaOption(pay_off, expiry=T)
    param_dict = helper.build_constant_market_param(0, 0, 0, 0)
    RNGenerator = helper.build_RNG("PSEUDO_RANDOM")
    prices = [10, 1, 0, 0]
    for i, S in enumerate(spots):
        EqModel = Sim.BSModel(
            spot=S,
            r=param_dict["Rate"],
            repo_rate=param_dict["Repo"],
            div_yield=param_dict["Div"],
            vol=param_dict["Vol"],
            RNG=RNGenerator,
        )
        collector = helper.build_collector("PATH_COUNT", 10)
        results = Pricer.vanilla_mc_pricer(
            option, param_dict["Rate"], EqModel, collector
        )
        analytic = Pricer.BS_PUT(S, K, T, 0, 1e-9)
        assert results.get_mean() == prices[i]
        assert abs(prices[i] - analytic) < 1e-7
        assert results.get_std_err() == 0.0
        assert results.get_path_count() == 10


def test_vanilla_call_pricer_zero_vol(tolerance=1e-8):
    K, maturity = 100, 5.0
    spots = [90, 100, 101, 110]
    pay_off = O.PayOffCall(strike=K)
    option = O.VanillaOption(pay_off, expiry=maturity)
    param_dict = helper.build_constant_market_param(
        rate=0.05, vol=0.0, repo=0.02, div=0.01
    )
    RNGenerator = helper.build_RNG("PSEUDO_RANDOM")
    discount_factor_r = np.exp(-param_dict["Rate"].get_mean(0, maturity) * maturity)
    discount_factor_q = np.exp(-param_dict["Repo"].get_mean(0, maturity) * maturity)
    discount_factor_d = np.exp(-param_dict["Div"].get_mean(0, maturity) * maturity)
    effective_DF = discount_factor_r / discount_factor_q / discount_factor_d
    terminal_payoff = [max(s / effective_DF - K, 0) for s in spots]
    discounted_payoff = [discount_factor_r * p for p in terminal_payoff]
    num_path = 10
    for i, S in enumerate(spots):
        EqModel = Sim.BSModel(
            spot=S,
            r=param_dict["Rate"],
            repo_rate=param_dict["Repo"],
            div_yield=param_dict["Div"],
            vol=param_dict["Vol"],
            RNG=RNGenerator,
        )
        collector = helper.build_collector("PATH_COUNT", num_path)
        results = Pricer.vanilla_mc_pricer(
            option, param_dict["Rate"], EqModel, collector
        )
        assert abs(results.get_mean() - discounted_payoff[i]) < tolerance
        assert results.get_std_err() == 0.0
        assert results.get_path_count() == num_path
