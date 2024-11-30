from abc import ABC, abstractmethod

class PlotterMap(ABC):
    @abstractmethod
    def plot(self, x, y):
        pass
