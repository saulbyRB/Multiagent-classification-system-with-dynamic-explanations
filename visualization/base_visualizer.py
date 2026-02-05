# visualization/base_visualizer.py

from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    """
    Interfaz base para visualización de explicaciones.
    """

    @abstractmethod
    def can_visualize(self, explanation: dict) -> bool:
        """
        Indica si este visualizador puede manejar esta explicación.
        """
        pass

    @abstractmethod
    def plot(self, explanation: dict, **kwargs):
        """
        Renderiza la explicación.
        """
        pass
