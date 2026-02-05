# visualization/shap_visualizer.py

import matplotlib.pyplot as plt
import numpy as np

from visualization.base_visualizer import BaseVisualizer

class ShapVisualizer(BaseVisualizer):

    def can_visualize(self, explanation: dict) -> bool:
        return (
            explanation["explainer"] == "shap"
            and explanation["scope"] == "local"
            and explanation["details"]["type"] == "feature_importance"
        )

    def plot(self, explanation: dict, max_features=15):
        values = np.array(explanation["details"]["values"])
        names = explanation["details"].get("feature_names")

        if names is None:
            names = [f"f{i}" for i in range(len(values))]

        idx = np.argsort(np.abs(values))[::-1][:max_features]
        values = values[idx]
        names = [names[i] for i in idx]

        plt.figure(figsize=(8, 4))
        plt.barh(names[::-1], values[::-1])
        plt.title("SHAP Feature Importance")
        plt.xlabel("Contribution")
        plt.tight_layout()
        plt.show()
