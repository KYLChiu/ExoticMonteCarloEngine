from ExoticEngine.Solvers import ImpliedVol as IV
from ExoticEngine.Solvers import solvers
import numpy as np


def test_bisection_solver(tolerance=1e-7):
    linear_f = lambda x: 2 * x + 6
    quadratic_f = lambda x: 3 * x ** 2 + 6 + x + 1
    exp_f = lambda x: 2. * np.exp(0.5 * x) - 100
    functions = [linear_f, quadratic_f, exp_f]
    inputs = [-2000.1, 5.2, -0.5]
    targets = [functions[i](x) for i, x in enumerate(inputs)]
    bounds = [[-3000, 0], [0, 50], [-10, 100]]
    solver_tolerance = tolerance*1.1
    assert tolerance < solver_tolerance
    for i, f in enumerate(functions):
        solver_result = solvers.bisection(f,
                                          targets[i],
                                          bounds[i][0],
                                          bounds[i][1],
                                          tolerance=solver_tolerance,
                                          max_iteration=50)
        assert abs(solver_result - inputs[i]) < tolerance
