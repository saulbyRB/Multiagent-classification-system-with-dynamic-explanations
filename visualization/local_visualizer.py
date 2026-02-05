# visualization/local_visualizer.py

from visualization.base_visualizer import BaseVisualizer
from visualization.shap_visualizer import ShapVisualizer
from visualization.lime_visualizer import LimeVisualizer

class LocalExplanationVisualizer:
    """
    Visualizador genérico para explicaciones locales.
    """

    def __init__(self):
        self.visualizers = [
            ShapVisualizer(),
            LimeVisualizer()
        ]

    def plot(self, explanation: dict, **kwargs):
        for viz in self.visualizers:
            if viz.can_visualize(explanation):
                return viz.plot(explanation, **kwargs)

        raise ValueError(
            f"No hay visualizador para el explainer '{explanation['explainer']}'"
        )
