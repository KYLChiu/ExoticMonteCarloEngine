import abc
import numbers
from typing import final

from ExoticEngine.Solvers import ImpliedVol as IV


class NumericalInversion(abc.ABC):
    def __init__(
        self,
        func_obj: IV.FunctionObject,
        target: float,
        start: tuple[float],
        tolerance: float = 1e-8,
        max_iteration: int = 30,
    ):
        assert tolerance >= 1e-13
        assert 2 < max_iteration < 100
        assert 0 < len(start) <= 2
        self._tolerance = tolerance
        self._max_iteration = max_iteration
        self._start = start
        self._target = target
        self._F = func_obj
        self._counter = 0

    @final
    def _eval_termination_condition(self, x):
        if self._counter >= self._max_iteration:
            print(
                f"WARNING: max number of iterations ({self._max_iteration}) reached!! \
                    target={self._target}, current={x}, tolerance={self._tolerance}"
            )
            return True
        else:
            return abs(x - self._target) < self._tolerance

    @abc.abstractmethod
    def solver(self):
        pass


# Bisection can be slow
@final
class Bisection(NumericalInversion):
    def __init__(
        self,
        func_obj: IV.FunctionObject,
        target: float,
        start: tuple[float],
        tolerance: float = 1e-8,
        max_iteration: int = 30,
    ):
        assert len(start) == 2
        super().__init__(func_obj, target, start, tolerance, max_iteration)

    def solver(self):
        """
        Assumes f: R -> R is a monotonically increasing function
        f(x) must only take 1 argument
        default max_iteration = 30
        let n=num_iteration: (upper_n - lower_n)/(upper_0 - lower_0) = 0.5^n
        """
        if self._start[0] < self._start[1]:
            lower, upper = self._start[0], self._start[1]
        elif self._start[0] > self._start[1]:
            upper, lower = self._start[0], self._start[1]
        else:
            raise Exception(
                f"Starting bounds cannot be the same: "
                f"lower={self._start[0]}, upper={self._start[1]}"
            )
        mid_point = 0.5 * (upper + lower)
        y = self._F.f(mid_point)
        assert self._F.f(upper) >= y >= self._F.f(lower)

        terminate = self._eval_termination_condition(y)
        while not terminate:
            if y < self._target:
                lower = mid_point
            elif y > self._target:
                upper = mid_point
            mid_point = 0.5 * (upper + lower)
            y = self._F.f(mid_point)
            self._counter += 1
            terminate = self._eval_termination_condition(y)
        return mid_point


# Newton-Raphson is fast,
# but requires first order derivative to be well-defined
@final
class NewtonRaphson(NumericalInversion):
    def __init__(
        self,
        func_obj: IV.FunctionObject,
        target: float,
        start: tuple[float],
        tolerance: float = 1e-8,
        max_iteration: int = 30,
    ):
        if type(start) == tuple:
            assert len(start) == 1
        else:
            assert isinstance(start, numbers.Number)
            start = (start,)
        super().__init__(func_obj, target, start, tolerance, max_iteration)
        assert self._F.f(self._start[0]) and self._F.derivative(self._start[0])

    def solver(self):
        """
        Assumes f: R -> R is a monotonically increasing function
        f(x) must only take 1 argument
        default max_iteration = 30
        """
        start = self._start[0]
        y = self._F.f(start)
        while not self._eval_termination_condition(y):
            gradient = self._F.derivative(start)
            # NR algorithm: x_new = x0 + (target - f(x0)) / f'(x0)
            start += (self._target - y) / gradient
            y = self._F.f(start)
            self._counter += 1
        return start
