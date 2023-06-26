from ExoticEngine.Statistics import RandomNumberGenerators as RNG
from ExoticEngine.MarketDataObject import Parameter as P
from ExoticEngine.Statistics import Statistics as Stats


def build_constant_market_param(rate: float, vol: float, repo: float, div: float):
    Rate = P.Parameter(param=lambda t: rate)
    Vol = P.Parameter(param=lambda t: vol)
    Repo = P.Parameter(param=lambda t: repo)
    Div = P.Parameter(param=lambda t: div)
    return {"Rate": Rate, "Vol": Vol, "Repo": Repo, "Div": Div}


def build_RNG(random_number_type: str):
    random_number = RNG.RandomNumberType(random_number_type)
    return RNG.TestRandom(random_number_type=random_number)


def build_collector(condition_type: str, criteria: float):
    condition = Stats.ConditionType(condition_type)
    termination_condition = Stats.TerminationCondition(condition, criteria)
    return Stats.GetStatistics(termination_condition)
