# visualization.py
"""
Módulo unificado de visualización:
- Resultados de métricas y predicciones
- Arquitectura de agentes
- Explicaciones (SHAP, LIME, Counterfactuals)
- Seguimiento temporal de explicaciones
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ===================== Resultados generales =====================

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
    values = [m[metric_name] for m in metrics_history]
    plt.figure(figsize=(6,4))
    plt.plot(values, marker='o')
    plt.title(f"{metric_name.capitalize()} Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)
    plt.show()

def plot_counterfactual(original, counterfactual, feature_names=None, title="Counterfactual"):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(original))]
    df = pd.DataFrame([original, counterfactual], index=["original", "counterfactual"], columns=feature_names)
    df.T.plot(kind='bar', figsize=(12,6))
    plt.title(title)
    plt.ylabel("Feature value")
    plt.show()

# ===================== Arquitectura de agentes =====================

def plot_agent_architecture(agents: dict):
    """
    agents: dict {agent_name: {'type': 'classifier/NN', 'connections': [other_agent_names]}}
    """
    G = nx.DiGraph()
    for agent_name, info in agents.items():
        G.add_node(agent_name, type=info.get('type', 'classifier'))
        for conn in info.get('connections', []):
            G.add_edge(agent_name, conn)

    pos = nx.spring_layout(G, seed=42)
    node_colors = ['skyblue' if G.nodes[n]['type']=='classifier' else 'lightgreen' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000,
            font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Agent Architecture")
    plt.show()

# ===================== Seguimiento temporal explicaciones =====================

def plot_explanation_similarity(agent):
    if not hasattr(agent, "explanation_history") or len(agent.explanation_history) < 2:
        print(f"[{agent.id}] No hay suficiente histórico de explicaciones")
        return
    sims = []
    history = agent.explanation_history
    for i in range(1, len(history)):
        sims.append(cosine_similarity(history[i-1].reshape(1,-1), history[i].reshape(1,-1))[0,0])
    plt.figure(figsize=(6,4))
    plt.plot(sims, marker='o')
    plt.title(f"{agent.id} - Similitud de explicaciones entre iteraciones")
    plt.xlabel("Iteración")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

def plot_explanation_divergence(agents):
    min_len = min(len(a.explanation_history) for a in agents.values())
    sims_over_time = []
    for i in range(min_len):
        vectors = np.array([a.explanation_history[i] for a in agents.values()])
        mean_vec = np.mean(vectors, axis=0)
        sims = [cosine_similarity(vec.reshape(1,-1), mean_vec.reshape(1,-1))[0,0] for vec in vectors]
        sims_over_time.append(np.mean(sims))
    plt.figure(figsize=(6,4))
    plt.plot(sims_over_time, marker='o')
    plt.title("Similitud promedio de explicaciones entre agentes")
    plt.xlabel("Iteración")
    plt.ylabel("Mean Cosine Similarity")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

# ===================== Local explainers =====================

class BaseVisualizer:
    """Clase base para explainers locales"""
    def can_visualize(self, explanation: dict) -> bool:
        raise NotImplementedError
    def plot(self, explanation: dict, **kwargs):
        raise NotImplementedError

class ShapVisualizer(BaseVisualizer):
    def can_visualize(self, explanation: dict) -> bool:
        return (explanation["explainer"]=="shap" and explanation["scope"]=="local")
    def plot(self, explanation: dict, max_features=15):
        values = np.array(explanation["details"]["values"])
        names = explanation["details"].get("feature_names", [f"f{i}" for i in range(len(values))])
        idx = np.argsort(np.abs(values))[::-1][:max_features]
        values = values[idx]
        names = [names[i] for i in idx]
        plt.figure(figsize=(8,4))
        plt.barh(names[::-1], values[::-1])
        plt.title("SHAP Feature Importance")
        plt.xlabel("Contribution")
        plt.tight_layout()
        plt.show()

class LimeVisualizer(BaseVisualizer):
    def can_visualize(self, explanation: dict) -> bool:
        return (explanation["explainer"]=="lime" and explanation["scope"]=="local")
    def plot(self, explanation: dict, max_features=10):
        weights = explanation["details"]["feature_weights"]
        items = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:max_features]
        names, values = zip(*items)
        plt.figure(figsize=(8,4))
        plt.barh(names[::-1], values[::-1])
        plt.title("LIME Feature Weights")
        plt.xlabel("Weight")
        plt.tight_layout()
        plt.show()

class LocalExplanationVisualizer:
    """Delegado de visualizadores locales"""
    def __init__(self):
        self.visualizers = [ShapVisualizer(), LimeVisualizer()]
    def plot(self, explanation: dict, **kwargs):
        for viz in self.visualizers:
            if viz.can_visualize(explanation):
                return viz.plot(explanation, **kwargs)
        raise ValueError(f"No hay visualizador para el explainer '{explanation['explainer']}'")

def plot_agents_dashboard(agents, metrics=["accuracy", "f1", "precision", "recall"]):
    """
    Dashboard por agente:
    - Evolución de métricas
    - Evolución de confianza
    - Similitud de explicaciones
    """
    n_agents = len(agents)
    fig, axes = plt.subplots(n_agents, 3, figsize=(18, 5*n_agents))

    if n_agents == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (agent_id, agent) in enumerate(agents.items()):
        # -------------------- Métricas --------------------
        ax_metrics = axes[idx,0]
        if hasattr(agent, "metrics_history") and agent.metrics_history:
            for m in metrics:
                values = [mh.get(m, 0.0) for mh in agent.metrics_history]
                ax_metrics.plot(values, marker='o', label=m.capitalize())
            ax_metrics.set_title(f"{agent_id} - Metrics over Iterations")
            ax_metrics.set_xlabel("Iteration")
            ax_metrics.set_ylabel("Score")
            ax_metrics.set_ylim(0,1)
            ax_metrics.grid(True)
            ax_metrics.legend()
        else:
            ax_metrics.text(0.5, 0.5, "No metrics", ha='center', va='center')
            ax_metrics.set_axis_off()

        # -------------------- Confianza --------------------
        ax_conf = axes[idx,1]
        if hasattr(agent, "metrics_history") and agent.metrics_history:
            if "confidence" in agent.metrics_history[0]:
                confs = [mh.get("confidence", np.nan) for mh in agent.metrics_history]
            else:
                confs = []
            if confs:
                ax_conf.plot(confs, marker='o', color='orange')
                ax_conf.set_ylim(0,1)
                ax_conf.set_title(f"{agent_id} - Confidence over Iterations")
                ax_conf.set_xlabel("Iteration")
                ax_conf.set_ylabel("Confidence")
                ax_conf.grid(True)
            else:
                ax_conf.text(0.5, 0.5, "No confidence", ha='center', va='center')
                ax_conf.set_axis_off()
        else:
            ax_conf.text(0.5, 0.5, "No confidence", ha='center', va='center')
            ax_conf.set_axis_off()

        # -------------------- Similitud de explicaciones --------------------
        ax_exp = axes[idx,2]
        if hasattr(agent, "explanation_history") and len(agent.explanation_history) > 1:
            sims = []
            history = agent.explanation_history
            for i in range(1, len(history)):
                sims.append(cosine_similarity(history[i-1].reshape(1,-1),
                                              history[i].reshape(1,-1))[0,0])
            ax_exp.plot(sims, marker='o', color='green')
            ax_exp.set_ylim(0,1)
            ax_exp.set_title(f"{agent_id} - Explanation similarity")
            ax_exp.set_xlabel("Iteration")
            ax_exp.set_ylabel("Cosine similarity")
            ax_exp.grid(True)
        else:
            ax_exp.text(0.5, 0.5, "No explanation history", ha='center', va='center')
            ax_exp.set_axis_off()

    plt.tight_layout()
    plt.show()
