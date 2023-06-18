import abc
import statistics as stat
from enum import Enum
import numpy as np


class TerminationCondition:
    def __init__(self, condition, criteria):
        """Try to make this more type safe"""
        self._condition = condition # this should be an enum
        self._criteria = criteria
        if self._condition == "PATH_COUNT":
            assert criteria % 1 == 0
        assert criteria > 0.

    def get_termination_criteria(self):
        return self._criteria

    def get_termination_condition(self):
        return self._condition

class GetStatistics:
    def __init__(self, TerminationCondition):
        self._running_sum = 0.
        self._paths_done = 0
        self._result_set = []
        self._TerminationCondition = TerminationCondition

    def add_one_result(self, new_result):
        assert len(self._result_set) == self._paths_done
        self._running_sum += new_result
        self._paths_done += 1
        self._result_set.append(new_result)

    def get_mean_so_far(self):
        return self._running_sum / self._paths_done

    def get_std_dev_so_far(self):
        if self._paths_done < 2:
            raise Exception("Path count is less than 2, path count = "+str(self._paths_done))
        return stat.stdev(self._result_set) # might not be fast?

    def get_std_err_so_far(self):
        return self.get_std_dev_so_far() / np.sqrt(self._paths_done)

    def get_path_count(self):
        return self._paths_done

    def get_pathwise_results(self):
        assert len(self._result_set) == self._paths_done
        return self._result_set

    def terminate(self, min_path=5) -> bool:
        condition = self._TerminationCondition.get_termination_condition()
        criteria = self._TerminationCondition.get_termination_criteria()
        if self._paths_done <= min_path:
            return False
        elif condition == "CONVERGENCE":
            return abs(self._running_sum - stat.mean(self._result_set[:-1])) <= criteria
        elif condition == "STANDARD_ERROR":
            return abs(self.get_std_dev_so_far()) <= criteria
        elif condition == "PATH_COUNT":
            if self._paths_done > criteria:
                print("WARNING! path count ("+str(self.get_path_count())+") > termination criteria ("+str(criteria)+")")
                return True
            return self._paths_done == criteria
        else:
            raise Exception("Should never happen...condition = "+condition)

