# visualization/lime_visualizer.py

import matplotlib.pyplot as plt
from visualization.base_visualizer import BaseVisualizer

class LimeVisualizer(BaseVisualizer):

    def can_visualize(self, explanation: dict) -> bool:
        return (
            explanation["explainer"] == "lime"
            and explanation["scope"] == "local"
            and explanation["details"]["type"] == "linear_approximation"
        )

    def plot(self, explanation: dict, max_features=10):
        weights = explanation["details"]["feature_weights"]

        items = sorted(
            weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_features]

        names, values = zip(*items)

        plt.figure(figsize=(8, 4))
        plt.barh(names[::-1], values[::-1])
        plt.title("LIME Feature Weights")
        plt.xlabel("Weight")
        plt.tight_layout()
        plt.show()
