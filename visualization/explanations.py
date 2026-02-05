import shap
import matplotlib.pyplot as plt
import pandas as pd

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
