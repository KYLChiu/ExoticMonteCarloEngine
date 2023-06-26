from ExoticEngine.Solvers import ImpliedVol as IV
from ExoticEngine.Solvers import solvers
from ExoticEngine import MonteCarloPricer as Pricer
import numpy as np


def test_bisection_solver(tolerance=1e-7):
    linear_f = lambda x: 2 * x + 6
    quadratic_f = lambda x: 3 * x ** 2 + 6 + x + 1
    exp_f = lambda x: 2. * np.exp(0.5 * x) - 100
    functions = [linear_f, quadratic_f, exp_f]
    inputs = [-2000.1, 5.2, -0.5]
    targets = [functions[i](x) for i, x in enumerate(inputs)]
    bounds = [[-3000, 0], [0, 50], [-10, 100]]
    solver_tolerance = tolerance * 0.9
    assert tolerance > solver_tolerance > 0
    for i, f in enumerate(functions):
        solver_result = solvers.bisection(f,
                                          targets[i],
                                          bounds[i][0],
                                          bounds[i][1],
                                          tolerance=solver_tolerance,
                                          max_iteration=50)
        assert abs(solver_result - inputs[i]) < tolerance


def test_implied_vol_bisection_solver(tolerance=1e-8):
    S, K, T, r, vol = 100, 105, 3., 0.01, 0.32
    BS = [IV.BSModel("CALL", S, K, T, r), IV.BSModel("PUT", S, K, T, r)]
    targets = [Pricer.BS_CALL(S, K, T, r, vol), Pricer.BS_PUT(S, K, T, r, vol)]
    bounds = [1e-5, 3.]
    solver_tolerance = tolerance * 0.9
    assert tolerance > solver_tolerance > 0
    for i, model in enumerate(BS):
        IV_result = solvers.bisection(lambda x: model.f(x),
                                      targets[i],
                                      bounds[0],
                                      bounds[1],
                                      tolerance=solver_tolerance,
                                      max_iteration=50)
        assert abs(IV_result - vol) < tolerance
