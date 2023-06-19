from ExoticEngine.Statistics import Statistics as Stats
import numpy as np


def test_path_count_termination():
    terminal_path = 10
    condition = Stats.ConditionType("PATH_COUNT")
    termination_condition = Stats.TerminationCondition(condition, terminal_path)
    collector = Stats.GetStatistics(termination_condition)
    terminate = False
    while not terminate:
        collector.add_one_result(1)
        terminate = collector.terminate(min_path=5)
    assert collector.get_path_count() == terminal_path
    assert len(collector.get_pathwise_results()) == terminal_path


def test_convergence_termination():
    convergence_tolerance = 0.99
    condition = Stats.ConditionType("CONVERGENCE")
    termination_condition = Stats.TerminationCondition(condition, convergence_tolerance)
    collector = Stats.GetStatistics(termination_condition)
    series = [10, 4, 1, 1, 0]
    check = [False, False, False, False, True]
    for i, s in enumerate(series):
        collector.add_one_result(s)
        terminate = collector.terminate(min_path=2)
        assert terminate == check[i]
    assert collector.get_path_count() == 5
    assert collector.get_pathwise_results() == series


def test_mean_value():
    terminal_path = 10
    condition = Stats.ConditionType("PATH_COUNT")
    termination_condition = Stats.TerminationCondition(condition, terminal_path)
    collector = Stats.GetStatistics(termination_condition)
    for i in range(terminal_path):
        collector.add_one_result(i)
        assert collector.get_pathwise_results()[i] == i
    assert collector.get_mean_so_far() == 4.5
