import numpy as np

from ExoticEngine.MarketDataObject import Parameter as P

# Parameter probably needs refactoring later


def test_mean(tolerance=1e-9):
    param = P.Parameter(lambda t: 3 * t**2)
    assert abs(param.get_mean(1.0, 5.0) - 124.0 / 4.0) < tolerance

    param = P.Parameter(lambda t: 2.0 * t)
    assert abs(param.get_mean(0.0, 2.0) - 2.0) < tolerance

    param = P.Parameter(lambda t: 87.0)
    assert abs(param.get_mean(15.0, 53.0) - 87.0) < tolerance


def test_root_mean_square(tolerance=1e-9):
    param = P.Parameter(lambda t: 3 * t)
    assert abs(param.get_root_mean_square(0, 2.0) - np.sqrt(12)) < tolerance

    param = P.Parameter(lambda t: t**0.5)
    assert abs(param.get_root_mean_square(1.0, 3.0) - np.sqrt(2)) < tolerance
