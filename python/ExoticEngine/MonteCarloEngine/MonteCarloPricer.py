import multiprocess
from copy import deepcopy
from abc import abstractmethod
from ExoticEngine.Statistics import Statistics as Stats

class MCPricer:

    @abstractmethod
    def __init__(self, result_collector: Stats.GetStatistics, num_cores: int):
        assert num_cores > 0
        self._collector = deepcopy(result_collector)
        self._num_cores = int(num_cores)

    @abstractmethod
    def _generate_path(self):
        pass

    def _generate_paths(self, res, num_paths: int):
       paths = [self._generate_path() for _ in range(num_paths)]
       res.extend(paths)

    @property 
    def stats(self):
        return self._collector
    
    def price(self):
        if self._num_cores == 1 or self._collector._termination_condition.get_termination_condition() != "PATH_COUNT":
            while not self._collector.terminate():
                result = [self._generate_path()]
                self._collector.add_results(result)
        else:
            # Only works for PATH_COUNT for now. Not clear how to terminate other ones cleanly and divide work correctly.
            ps = []
            manager = multiprocess.Manager()
            results = manager.list()
            num_paths_per_process = int(self._collector._termination_condition.get_termination_criteria() / self._num_cores)
            for i in range(self._num_cores):
                if i == self._num_cores - 1:
                    # Force termination.
                    num_paths_per_process = int(num_paths_per_process) + int(self._collector._termination_condition.get_termination_criteria()) % self._num_cores
                p = multiprocess.Process(target=self._generate_paths, args=(results, num_paths_per_process))
                ps.append(p)
                p.start()
            for p in ps:
                p.join()
            self._collector.add_results(list(results))
                
                


    
