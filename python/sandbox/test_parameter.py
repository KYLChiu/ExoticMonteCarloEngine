from ExoticEngine.MarketDataObject import Parameter as P
import numpy as np


# Parameter probably needs refactoring later

def test_mean(tolerance=1e-9):
    param = P.Parameter(lambda t: 3 * t ** 2)
    assert abs(param.get_mean(1., 5.) - 124. / 4.) < tolerance

    param = P.Parameter(lambda t: 2. * t)
    assert abs(param.get_mean(0., 2.) - 2.) < tolerance

    param = P.Parameter(lambda t: 87.)
    assert abs(param.get_mean(15., 53.) - 87.) < tolerance


def test_root_mean_square(tolerance=1e-9):
    param = P.Parameter(lambda t: 3 * t)
    assert abs(param.get_root_mean_square(0, 2.) - np.sqrt(12)) < tolerance

    param = P.Parameter(lambda t: t ** 0.5)
    assert abs(param.get_root_mean_square(1., 3.) - np.sqrt(2)) < tolerance
