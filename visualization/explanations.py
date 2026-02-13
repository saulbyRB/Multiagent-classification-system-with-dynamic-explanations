import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def plot_shap_summary(shap_values, feature_names=None, title="SHAP Summary"):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(shap_values.values.shape[1])]
    shap.summary_plot(shap_values, feature_names=feature_names, show=True, plot_type="bar", max_display=15, plot_size=(10,5))
    
def plot_shap_instance(shap_values, instance_idx=0, feature_names=None, title="SHAP Explanation"):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(shap_values.values.shape[1])]
    shap.plots.waterfall(shap_values[instance_idx])

def plot_lime_explanation(lime_exp, feature_names=None, title="LIME Explanation"):
    lime_exp.show_in_notebook(show_table=True)

def plot_counterfactual(original, counterfactual, feature_names=None, title="Counterfactual"):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(original))]
    df = pd.DataFrame([original, counterfactual], index=["original", "counterfactual"], columns=feature_names)
    df.T.plot(kind='bar', figsize=(12,6))
    plt.title(title)
    plt.ylabel("Feature value")
    plt.show()

def plot_explanation_similarity(agent):
    """
    Muestra cómo cambian las explicaciones de un agente a lo largo del tiempo
    usando similitud coseno entre iteraciones consecutivas
    """
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
    """
    Muestra la divergencia entre agentes en cada iteración
    usando la similitud coseno promedio
    """
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