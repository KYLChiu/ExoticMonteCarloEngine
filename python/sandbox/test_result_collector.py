from ExoticEngine.Statistics import Statistics as Stats


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


def test_mean_value():
    terminal_path = 10
    condition = Stats.ConditionType("PATH_COUNT")
    termination_condition = Stats.TerminationCondition(condition, terminal_path)
    collector = Stats.GetStatistics(termination_condition)
    for i in range(terminal_path):
        collector.add_one_result(i)
        assert collector.get_pathwise_results()[i] == i
    assert collector.get_mean_so_far() == 4.5
