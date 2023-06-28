import abc
import statistics as stat
from enum import Enum
import numpy as np
from typing import final


# from typing import Union # Union type check only in python10


class ConditionType(Enum):
    PATH_COUNT = "PATH_COUNT"
    STANDARD_ERROR = "STANDARD_ERROR"
    CONVERGENCE = "CONVERGENCE"


@final
class TerminationCondition:
    def __init__(self, condition: ConditionType, criteria: float):
        self._condition = condition.value
        self._criteria = criteria
        if self._condition == "PATH_COUNT":
            assert isinstance(criteria, int)
        else:
            assert isinstance(criteria, float)
        assert criteria > 0.

    def get_termination_criteria(self) -> float:
        return self._criteria

    def get_termination_condition(self) -> str:
        return self._condition


@final
class GetStatistics:
    def __init__(self, termination_condition: TerminationCondition):
        self._running_sum: float = 0.
        self._prev_running_sum: float = 0.
        self._paths_done: int = 0
        self._result_set: list[float] = []
        self._termination_condition: TerminationCondition = termination_condition

    def add_results(self, new_results: list[float]) -> None:
        assert len(self._result_set) == self._paths_done
        self._prev_running_sum = self._running_sum
        self._running_sum += sum(new_results)
        self._paths_done += len(new_results)
        self._result_set.extend(new_results)

    def get_mean(self) -> float:
        return self._running_sum / self._paths_done

    def get_std_dev(self) -> float:
        if self._paths_done < 2:
            raise Exception(f"Path count is less than 2, path count = {self._paths_done}")
        return stat.stdev(self._result_set)  # might not be fast?

    def get_std_err(self) -> float:
        return self.get_std_dev() / np.sqrt(self._paths_done)

    def get_path_count(self) -> int:
        return self._paths_done

    def get_pathwise_results(self) -> list[float]:
        assert len(self._result_set) == self._paths_done
        return self._result_set

    def terminate(self, min_path: int = 5, max_path: int = 25000) -> bool:
        assert min_path >= 2
        assert max_path > min_path
        condition = self._termination_condition.get_termination_condition()
        criteria = self._termination_condition.get_termination_criteria()
        if self._paths_done <= min_path:
            return False
        elif self._paths_done >= max_path:
            print(f"WARNING! Maximum number of paths reached: path_count={self.get_path_count()}, max_path={max_path}")
            return True
        elif condition == "CONVERGENCE":
            return abs(self.get_mean() - self._prev_running_sum / (self.get_path_count() - 1)) <= criteria
        elif condition == "STANDARD_ERROR":
            return abs(self.get_std_err()) <= criteria
        elif condition == "PATH_COUNT":
            if self._paths_done > criteria:
                print(f"WARNING! path_count ({self.get_path_count()}) > termination criteria {criteria})")
                return True
            return self._paths_done == criteria
        else:
            raise Exception(f"Should never happen...condition = {condition}")
