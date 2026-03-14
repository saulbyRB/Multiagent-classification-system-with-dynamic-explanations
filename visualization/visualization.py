# visualization/visualization.py
"""
Módulo unificado de visualización:
- Resultados de métricas y predicciones
- Arquitectura de agentes
- Explicaciones (SHAP, LIME, Counterfactuals)
- Seguimiento temporal de explicaciones
- Dashboard multi-agente con métricas explicativas
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


# ===================== Resultados generales =====================

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.show()


def plot_class_probabilities(probas, labels=None, instance_idx=0,
                              title="Predicted Probabilities"):
    prob = probas[instance_idx]
    if labels is None:
        labels = [f"Class {i}" for i in range(len(prob))]
    sns.barplot(x=labels, y=prob)
    plt.title(f"{title} - Instance {instance_idx}")
    plt.ylim(0, 1)
    plt.show()


def plot_metrics_over_time(metrics_history, metric_name="accuracy"):
    values = [m[metric_name] for m in metrics_history]
    plt.figure(figsize=(6, 4))
    plt.plot(values, marker='o')
    plt.title(f"{metric_name.capitalize()} Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)
    plt.show()


def plot_counterfactual(original, counterfactual, feature_names=None,
                        title="Counterfactual"):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(original))]
    df = pd.DataFrame(
        [original, counterfactual],
        index=["original", "counterfactual"],
        columns=feature_names
    )
    df.T.plot(kind='bar', figsize=(12, 6))
    plt.title(title)
    plt.ylabel("Feature value")
    plt.show()


# ===================== Arquitectura de agentes =====================

def plot_agent_architecture(agents: dict):
    """
    agents: dict {agent_name: {'type': 'classifier/NN',
                               'connections': [other_agent_names]}}
    """
    G = nx.DiGraph()
    for agent_name, info in agents.items():
        G.add_node(agent_name, type=info.get('type', 'classifier'))
        for conn in info.get('connections', []):
            G.add_edge(agent_name, conn)

    pos = nx.spring_layout(G, seed=42)
    node_colors = [
        'skyblue' if G.nodes[n]['type'] == 'classifier' else 'lightgreen'
        for n in G.nodes()
    ]
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Agent Architecture")
    plt.show()


# ===================== Seguimiento temporal de explicaciones =====================

def plot_explanation_similarity(agent):
    if not hasattr(agent, "explanation_history") or \
            len(agent.explanation_history) < 2:
        print(f"[{agent.id}] No hay suficiente histórico de explicaciones")
        return
    sims = []
    history = agent.explanation_history
    for i in range(1, len(history)):
        sims.append(
            cosine_similarity(
                history[i - 1].reshape(1, -1),
                history[i].reshape(1, -1)
            )[0, 0]
        )
    plt.figure(figsize=(6, 4))
    plt.plot(sims, marker='o')
    plt.title(f"{agent.id} - Similitud de explicaciones entre iteraciones")
    plt.xlabel("Iteración")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def plot_explanation_divergence(agents):
    agents = {
        k: a for k, a in agents.items()
        if hasattr(a, "explanation_history") and len(a.explanation_history) > 0
    }
    if len(agents) < 2:
        print("[Viz] No hay suficientes agentes con histórico de explicaciones")
        return

    min_len = min(len(a.explanation_history) for a in agents.values())
    sims_over_time = []
    for i in range(min_len):
        vectors  = np.array([a.explanation_history[i] for a in agents.values()])
        mean_vec = np.mean(vectors, axis=0)
        sims = [
            cosine_similarity(vec.reshape(1, -1), mean_vec.reshape(1, -1))[0, 0]
            for vec in vectors
        ]
        sims_over_time.append(np.mean(sims))

    plt.figure(figsize=(6, 4))
    plt.plot(sims_over_time, marker='o')
    plt.title("Similitud promedio de explicaciones entre agentes")
    plt.xlabel("Iteración")
    plt.ylabel("Mean Cosine Similarity")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


# ===================== Local explainers =====================

class BaseVisualizer:
    def can_visualize(self, explanation: dict) -> bool:
        raise NotImplementedError

    def plot(self, explanation: dict, **kwargs):
        raise NotImplementedError


class ShapVisualizer(BaseVisualizer):
    def can_visualize(self, explanation: dict) -> bool:
        return (explanation["explainer"] == "shap"
                and explanation["scope"] == "local")

    def plot(self, explanation: dict, max_features=15):
        values = np.array(explanation["details"]["values"])
        names  = explanation["details"].get(
            "feature_names", [f"f{i}" for i in range(len(values))]
        )
        idx    = np.argsort(np.abs(values))[::-1][:max_features]
        values = values[idx]
        names  = [names[i] for i in idx]
        plt.figure(figsize=(8, 4))
        plt.barh(names[::-1], values[::-1])
        plt.title("SHAP Feature Importance")
        plt.xlabel("Contribution")
        plt.tight_layout()
        plt.show()


class LimeVisualizer(BaseVisualizer):
    def can_visualize(self, explanation: dict) -> bool:
        return (explanation["explainer"] == "lime"
                and explanation["scope"] == "local")

    def plot(self, explanation: dict, max_features=10):
        weights = explanation["details"]["feature_weights"]
        items   = sorted(weights.items(), key=lambda x: abs(x[1]),
                         reverse=True)[:max_features]
        names, values = zip(*items)
        plt.figure(figsize=(8, 4))
        plt.barh(names[::-1], values[::-1])
        plt.title("LIME Feature Weights")
        plt.xlabel("Weight")
        plt.tight_layout()
        plt.show()


class LocalExplanationVisualizer:
    def __init__(self):
        self.visualizers = [ShapVisualizer(), LimeVisualizer()]

    def plot(self, explanation: dict, **kwargs):
        for viz in self.visualizers:
            if viz.can_visualize(explanation):
                return viz.plot(explanation, **kwargs)
        raise ValueError(
            f"No hay visualizador para el explainer '{explanation['explainer']}'"
        )


# ===================== Dashboard clásico por agente =====================

def plot_agents_dashboard(agents,
                          metrics=("accuracy", "f1", "precision", "recall")):
    """
    Dashboard por agente: métricas, confianza y similitud de explicaciones.
    """
    n_agents = len(agents)
    fig, axes = plt.subplots(n_agents, 3, figsize=(18, 5 * n_agents))

    if n_agents == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (agent_id, agent) in enumerate(agents.items()):

        # — Métricas —
        ax = axes[idx, 0]
        if hasattr(agent, "metrics_history") and agent.metrics_history:
            for m in metrics:
                values = [mh.get(m, 0.0) for mh in agent.metrics_history]
                ax.plot(values, marker='o', label=m.capitalize())
            ax.set_title(f"{agent_id} - Metrics over Iterations")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No metrics", ha='center', va='center')
            ax.set_axis_off()

        # — Confianza —
        ax = axes[idx, 1]
        confs = []
        if hasattr(agent, "metrics_history") and agent.metrics_history:
            if "confidence" in agent.metrics_history[0]:
                confs = [mh.get("confidence", np.nan)
                         for mh in agent.metrics_history]
        if confs:
            ax.plot(confs, marker='o', color='orange')
            ax.set_ylim(0, 1)
            ax.set_title(f"{agent_id} - Confidence over Iterations")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Confidence")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No confidence", ha='center', va='center')
            ax.set_axis_off()

        # — Similitud de explicaciones —
        ax = axes[idx, 2]
        if hasattr(agent, "explanation_history") and \
                len(agent.explanation_history) > 1:
            sims = []
            history = agent.explanation_history
            for i in range(1, len(history)):
                sims.append(
                    cosine_similarity(
                        history[i - 1].reshape(1, -1),
                        history[i].reshape(1, -1)
                    )[0, 0]
                )
            ax.plot(sims, marker='o', color='green')
            ax.set_ylim(0, 1)
            ax.set_title(f"{agent_id} - Explanation similarity")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cosine similarity")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No explanation history", ha='center',
                    va='center')
            ax.set_axis_off()

    plt.tight_layout()
    plt.show()


# ===================== Dashboard explicativo multi-agente =====================

def plot_explainability_dashboard(df: pd.DataFrame,
                                  classifier_ids: list,
                                  hard_idx: int,
                                  true_label: int):
    """
    Dashboard con 4 subplots por agente a partir del DataFrame de resultados:

    Col 0 — Accuracy vs Exp Quality
    Col 1 — Consensus & Stability
    Col 2 — Fidelity SHAP vs LIME vs Media
    Col 3 — Predicción & Acuerdo SHAP=LIME

    Parameters
    ----------
    df            : DataFrame generado por build_results_dataframe()
    classifier_ids: lista ordenada de agent_ids
    hard_idx      : índice de la instancia frontera (para el título)
    true_label    : clase real de la instancia (para el título)
    """
    n_agents = len(classifier_ids)
    fig = plt.figure(figsize=(18, 5 * n_agents))
    fig.suptitle(
        f"Evolución de explicaciones — instancia frontera idx={hard_idx} "
        f"(clase real={true_label})",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(n_agents, 4, figure=fig, hspace=0.45, wspace=0.35)

    for row_i, agent_id in enumerate(classifier_ids):
        sub   = df[df["agent"] == agent_id].reset_index(drop=True)
        iters = sub["iteration"]

        # Col 0: Accuracy vs Exp Quality
        ax = fig.add_subplot(gs[row_i, 0])
        ax.plot(iters, sub["accuracy"],    label="Accuracy",
                marker="o", markersize=3)
        ax.plot(iters, sub["exp_quality"], label="Exp quality",
                marker="s", markersize=3, linestyle="--")
        ax.set_title(f"{agent_id} — Accuracy vs Exp Quality")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Iteración")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 1: Consensus & Stability
        ax = fig.add_subplot(gs[row_i, 1])
        ax.plot(iters, sub["exp_consensus"], label="Consensus",
                marker="o", markersize=3)
        ax.plot(iters, sub["exp_stability"], label="Stability",
                marker="s", markersize=3, linestyle="--")
        ax.set_title(f"{agent_id} — Consensus & Stability")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Iteración")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 2: Fidelity SHAP vs LIME vs Media
        ax = fig.add_subplot(gs[row_i, 2])
        ax.plot(iters, sub["shap_fidelity"], label="SHAP",
                marker="o", markersize=3, color="steelblue")
        ax.plot(iters, sub["lime_fidelity"], label="LIME",
                marker="s", markersize=3, color="coral", linestyle="--")
        ax.plot(iters, sub["exp_fidelity"],  label="Mean",
                marker="^", markersize=3, color="green", linestyle=":")
        ax.set_title(f"{agent_id} — Fidelity SHAP vs LIME")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Iteración")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 3: Predicción & Acuerdo SHAP=LIME
        ax = fig.add_subplot(gs[row_i, 3])
        ax.step(iters, sub["prediction"],
                label="Predicción", where="mid", color="purple")
        ax.step(iters, sub["shap_lime_agree"].astype(int),
                label="SHAP=LIME",  where="mid", color="orange",
                linestyle="--")
        ax.set_title(f"{agent_id} — Predicción & Acuerdo")
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel("Iteración")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.show()


# ===================== Construcción del DataFrame de resultados =====================

def build_results_dataframe(aggregator_history: list,
                             classifier_ids: list,
                             feature_names: list) -> pd.DataFrame:
    """
    Construye el DataFrame de resultados a partir del historial del agregador.

    Parameters
    ----------
    aggregator_history : aggregator.global_history
    classifier_ids     : lista ordenada de agent_ids
    feature_names      : nombres de features del dataset

    Returns
    -------
    pd.DataFrame ordenado por (agent, iteration)
    """

    def _top_feature(explanation):
        if explanation is None:
            return None
        try:
            values  = np.array(explanation["details"]["values"])
            idx     = int(np.argmax(np.abs(values)))
            if feature_names and idx < len(feature_names):
                return feature_names[idx]
            return f"f{idx}"
        except Exception:
            return None

    rows = []
    for entry in aggregator_history:
        iteration     = entry["iteration"]
        evaluation    = entry["evaluation"]
        exp_detail    = evaluation["components"]["exp_detail"]
        per_explainer = exp_detail.get("per_explainer", {})
        decisions     = entry["decisions"]

        for idx, agent_id in enumerate(classifier_ids):
            explanations = entry["responses"][idx].get("explanations", [])

            shap_exp = next(
                (e for e in explanations if e.get("explainer") == "shap"), None
            )
            lime_exp = next(
                (e for e in explanations if e.get("explainer") == "lime"), None
            )
            shap_top = _top_feature(shap_exp)
            lime_top = _top_feature(lime_exp)

            def _get(name, metric):
                arr = per_explainer.get(name, {}).get(metric, [])
                return arr[idx] if idx < len(arr) else None

            rows.append({
                "agent":            agent_id,
                "iteration":        iteration,
                "accuracy":         entry["responses"][idx]["metrics"]["accuracy"],
                "f1":               entry["responses"][idx]["metrics"]["f1"],
                "precision":        entry["responses"][idx]["metrics"]["precision"],
                "recall":           entry["responses"][idx]["metrics"]["recall"],
                "prediction":       entry["responses"][idx]["prediction"],
                "confidence":       entry["responses"][idx].get("confidence"),
                "exp_quality":      exp_detail["quality"][idx],
                "exp_consensus":    exp_detail["consensus"][idx],
                "exp_stability":    exp_detail["stability"][idx],
                "exp_fidelity":     exp_detail["fidelity"][idx],
                "exp_agreement":    exp_detail["agreement"][idx],
                "shap_consensus":   _get("shap", "consensus"),
                "shap_stability":   _get("shap", "stability"),
                "shap_fidelity":    _get("shap", "fidelity"),
                "lime_consensus":   _get("lime", "consensus"),
                "lime_stability":   _get("lime", "stability"),
                "lime_fidelity":    _get("lime", "fidelity"),
                "shap_top_feature": shap_top,
                "lime_top_feature": lime_top,
                "shap_lime_agree":  (shap_top == lime_top
                                     and shap_top is not None),
                "decision":         decisions[idx],
            })

    return pd.DataFrame(rows).sort_values(
        ["agent", "iteration"]
    ).reset_index(drop=True)


# ===================== Resúmenes textuales =====================

def print_experiment_summary(df: pd.DataFrame,
                              classifier_ids: list,
                              feature_names: list):
    """
    Imprime todos los resúmenes estadísticos del experimento.
    """
    print("\n[Resumen] Medias por agente:")
    print(df.groupby("agent")[[
        "accuracy", "f1", "exp_quality", "exp_consensus",
        "exp_stability", "exp_fidelity", "exp_agreement"
    ]].mean().round(3))

    print("\n[Resumen] Evolución media por iteración (primeras 10):")
    print(df.groupby("iteration")[[
        "accuracy", "exp_quality", "exp_fidelity", "exp_agreement"
    ]].mean().round(3).head(10))

    print("\n[Resumen] Distribución de decisiones por agente:")
    print(df.groupby(["agent", "decision"]).size().unstack(fill_value=0))

    print("\n[Resumen] Acuerdo SHAP vs LIME por agente (% iteraciones):")
    print(df.groupby("agent")["shap_lime_agree"].mean().round(3))

    print("\n[Resumen] Fidelity media SHAP vs LIME por agente:")
    print(df.groupby("agent")[["shap_fidelity", "lime_fidelity"]].mean().round(3))

    print("\n[Resumen] Feature más votada por SHAP y LIME:")
    for agent_id in classifier_ids:
        sub = df[df["agent"] == agent_id]
        print(f"\n  {agent_id}:")
        print(f"    SHAP top: {sub['shap_top_feature'].value_counts().head(3).to_dict()}")
        print(f"    LIME top: {sub['lime_top_feature'].value_counts().head(3).to_dict()}")

    print("\n[Análisis] Correlación Accuracy ↔ Exp Quality por agente:")
    for agent_id in classifier_ids:
        sub  = df[df["agent"] == agent_id]
        corr = sub["accuracy"].corr(sub["exp_quality"])
        print(f"  {agent_id}: r={corr:.3f}")

    print("\n[Análisis] Predicción por agente e iteración:")
    print(df.pivot(index="iteration", columns="agent",
                   values="prediction").to_string())

    print("\n[Análisis] Estabilización de predicción:")
    for agent_id in classifier_ids:
        preds   = df[df["agent"] == agent_id]["prediction"].values
        changes = np.where(np.diff(preds) != 0)[0]
        if len(changes) == 0:
            print(f"  {agent_id}: estable desde iter 0 → pred={preds[0]}")
        else:
            print(f"  {agent_id}: último cambio en iter {changes[-1]+1}"
                  f" → pred final={preds[-1]}")

    print("\n[Análisis] Correlación SHAP consensus ↔ LIME consensus por agente:")
    for agent_id in classifier_ids:
        sub  = df[df["agent"] == agent_id]
        corr = sub["shap_consensus"].corr(sub["lime_consensus"])
        print(f"  {agent_id}: r={corr:.3f}")

