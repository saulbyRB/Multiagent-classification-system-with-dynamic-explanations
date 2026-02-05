import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.show()

def plot_class_probabilities(probas, labels=None, instance_idx=0, title="Predicted Probabilities"):
    prob = probas[instance_idx]
    if labels is None:
        labels = [f"Class {i}" for i in range(len(prob))]
    sns.barplot(x=labels, y=prob)
    plt.title(f"{title} - Instance {instance_idx}")
    plt.ylim(0,1)
    plt.show()

def plot_metrics_over_time(metrics_history, metric_name="accuracy"):
    """
    metrics_history: list of dicts {'accuracy':.., 'f1':.., ...}
    """
    values = [m[metric_name] for m in metrics_history]
    plt.plot(values, marker='o')
    plt.title(f"{metric_name} over time")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.show()
