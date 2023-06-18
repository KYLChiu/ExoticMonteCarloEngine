import abc


class CashFlow(abc.ABC):
    @abc.abstractmethod
    def get_cashflow(self):
        pass


