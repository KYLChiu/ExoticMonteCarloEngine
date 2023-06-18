import unittest

import numpy as np
import matplotlib.pyplot as plt

from ExoticEngine.Statistics import Statistics as Stats

def test_path_count_termination(terminal_path=10):
    TerminationCondition = Stats.TerminationCondition("PATH_COUNT", terminal_path)
    Collector = Stats.GetStatistics(TerminationCondition)
    terminate = False
    while not terminate:
        Collector.add_one_result(1)
        terminate = Collector.terminate(min_path=5)
    assert Collector.get_path_count() == terminal_path
    assert len(Collector.get_pathwise_results()) == terminal_path


def test_mean_value(terminal_path=10):
    TerminationCondition = Stats.TerminationCondition("PATH_COUNT", terminal_path)
    Collector = Stats.GetStatistics(TerminationCondition)
    for i in range(terminal_path):
        Collector.add_one_result(i)
        assert Collector.get_pathwise_results()[i] == i
    assert Collector.get_mean_so_far() == 4.5