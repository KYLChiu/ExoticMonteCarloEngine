import abc
import statistics as stat
from enum import Enum
import numpy as np
from typing import final
from typing import Union


class ConditionType(Enum):
    PATH_COUNT = "PATH_COUNT"
    STANDARD_ERROR = "STANDARD_ERROR"
    CONVERGENCE = "CONVERGENCE"


@final
class TerminationCondition:
    def __init__(self, condition: type[ConditionType], criteria: float | int):
        self._condition = condition.value
        self._criteria = criteria
        if self._condition == "PATH_COUNT":
            assert criteria % 1 == 0
            self._criteria = int(criteria)
        assert self._criteria > 0.

    def get_termination_criteria(self) -> Union[float, int]:
        return self._criteria

    def get_termination_condition(self):
        return self._condition


@final
class GetStatistics:
    def __init__(self, termination_condition):  # mypy error if termination_condition: type[TerminationCondition]
        self._running_sum: float = 0.
        self._paths_done: int = 0
        self._result_set: list[float] = []
        self._termination_condition = termination_condition

    def add_one_result(self, new_result) -> None:
        assert len(self._result_set) == self._paths_done
        self._running_sum += new_result
        self._paths_done += 1
        self._result_set.append(new_result)

    def get_mean_so_far(self) -> float:
        return self._running_sum / self._paths_done

    def get_std_dev_so_far(self) -> float:
        if self._paths_done < 2:
            raise Exception("Path count is less than 2, path count = " + str(self._paths_done))
        return stat.stdev(self._result_set)  # might not be fast?

    def get_std_err_so_far(self) -> float:
        return self.get_std_dev_so_far() / np.sqrt(self._paths_done)

    def get_path_count(self) -> int:
        return self._paths_done

    def get_pathwise_results(self) -> list[float]:
        assert len(self._result_set) == self._paths_done
        return self._result_set

    def terminate(self, min_path: int = 5) -> bool:
        assert min_path >= 2
        condition = self._termination_condition.get_termination_condition()
        criteria = self._termination_condition.get_termination_criteria()
        if self._paths_done <= min_path:
            return False
        elif condition == "CONVERGENCE":
            return abs(self._running_sum - stat.mean(self._result_set[:-1])) <= criteria
        elif condition == "STANDARD_ERROR":
            return abs(self.get_std_dev_so_far()) <= criteria
        elif condition == "PATH_COUNT":
            if self._paths_done > criteria:
                print("WARNING! path count (" + str(self.get_path_count()) + ") > termination criteria (" + str(
                    criteria) + ")")
                return True
            return self._paths_done == criteria
        else:
            raise Exception("Should never happen...condition = " + condition)
