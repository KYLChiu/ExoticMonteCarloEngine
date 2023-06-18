import abc
import numpy as np

class RandomBase(abc.ABC):
    @abc.abstractmethod
    def get_uniforms(self):
        pass

    @abc.abstractmethod
    def set_seed(self):
        pass

    def get_gaussian(self) -> float:
        """
        Cheating right now - using python package
        returns a single standard normal rv
        """
        return np.random.normal()


class TestRandom(RandomBase):
    def __init__(self, random_number_type):
        """Dummy class"""
        self._random_number_type = random_number_type

    def get_uniforms(self):
        raise Exception("Not implemented")

    def set_seed(self):
        raise Exception("Not implemented")