# run_agents.py
import asyncio

import sys
from pathlib import Path

# Asumiendo que notebooks/ está dentro de la raíz del proyecto
root_path = Path().resolve().parent  # sube un nivel
sys.path.append(str(root_path))


from agents.classifier_agent import ClassifierAgent
from agents.aggregator_agent import AggregatorAgent
from data.dataset_registry import DatasetRegistry
from data.loaders.sklearn_loader import SklearnLoader
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from explainers.shap_explainer import ShapExplainer

# ------------------- Dataset -------------------
registry = DatasetRegistry()
dataset_id = "iris"
registry.register(dataset_id, SklearnLoader(load_iris))
X, y, meta = registry.load(dataset_id)
print(f"[Dataset] Cargado {dataset_id}: {X.shape[0]} muestras, {X.shape[1]} features")

# ------------------- Clasificadores -------------------
classifier_jids = ["clf1@localhost", "clf2@localhost", "clf3@localhost"]
classifiers = []

for jid in classifier_jids:
    agent = ClassifierAgent(
        jid=jid,
        password="password",
        model=RandomForestClassifier(n_estimators=10),
        explainers=[ShapExplainer()],
        dataset_id=dataset_id,
        registry=registry
    )
    classifiers.append(agent)
    print(f"[ClassifierAgent] Preparado: {jid} con 1 explainer")

# ------------------- Agregador -------------------
aggregator = AggregatorAgent(
    jid="aggregator@localhost",
    password="password",
    classifier_jids=classifier_jids,
    max_iterations=2
)
aggregator.set_instance(X[:1])
print("[AggregatorAgent] Instancia asignada para evaluación")

# ------------------- Ejecución con logs -------------------
async def main():
    print("[Run] Iniciando agentes...")
    # Iniciar clasificadores y agregador
    await asyncio.gather(*(agent.start(auto_register=True) for agent in classifiers))
    await aggregator.start(auto_register=True)
    print("[Run] Agentes iniciados")

    # Lanzar primera iteración
    print("[Run] Lanzando primera iteración")
    await aggregator.start_iteration()

    # Esperar tiempo suficiente para procesar varias iteraciones
    total_wait = 15
    for i in range(total_wait):
        print(f"[Run] Esperando... {i+1}/{total_wait} seg")
        await asyncio.sleep(1)

    # Detener agentes
    print("[Run] Deteniendo agentes...")
    await asyncio.gather(*(agent.stop() for agent in classifiers))
    await aggregator.stop()
    print("[Run] Todos los agentes detenidos")

# ================= Ejecutar script =================
if __name__ == "__main__":
    asyncio.run(main())
