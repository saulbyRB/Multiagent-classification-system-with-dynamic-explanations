# agents/classifier_agent.py
import json
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ClassifierAgent(Agent):
    """
    Agente clasificador con:
    - Entrenamiento autónomo
    - Optimización interna
    - Recepción de feedback para ajustar hiperparámetros
    """

    def __init__(self, jid, password, model, explainer, dataset_id=None, registry=None, test_size=0.2, **kwargs):
        super().__init__(jid, password, **kwargs)
        self.model = model
        self.explainer = explainer
        self.dataset_id = dataset_id
        self.registry = registry
        self.test_size = test_size

        # Datos de prueba
        self.X_test = None
        self.y_test = None

        # Historial de métricas
        self.metrics_history = []

    async def setup(self):
        # Behaviour de entrenamiento y auto-optimización
        self.add_behaviour(self.TrainingBehaviour())

        # Behaviour cíclico de clasificación y feedback
        self.add_behaviour(self.ClassificationBehaviour())
        print(f"[{self.jid}] Agente clasificador listo.")

    # ==================== Training Behaviour ====================
    class TrainingBehaviour(OneShotBehaviour):
        async def run(self):
            if self.agent.dataset_id and self.agent.registry:
                X, y, meta = self.agent.registry.load(self.agent.dataset_id)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.agent.test_size)

                # Entrenamiento inicial con optimización simple
                self.agent.model.fit(X_train, y_train)
                self.agent.X_test = X_test
                self.agent.y_test = y_test

                # Evaluar y guardar métricas
                acc = accuracy_score(y_test, self.agent.model.predict(X_test))
                self.agent.metrics_history.append({"accuracy": acc})
                print(f"[{self.agent.jid}] Modelo entrenado. Accuracy inicial: {acc:.3f}")

                # Autoajuste simple
                self.agent.auto_tune_best(X_train, y_train, X_test, y_test)

    # ==================== Classification Behaviour ====================
    class ClassificationBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg is None:
                return

            try:
                content = json.loads(msg.body)
                action = content.get("action")

                if action == "classify":
                    X = content.get("data", self.agent.X_test)
                    instance_id = content.get("instance_id", 0)

                    y_pred = int(self.agent.model.predict(X)[instance_id])
                    confidence = self.agent.model.get_confidence(X)
                    confidence = float(confidence[instance_id]) if confidence is not None else None

                    explanation = self.agent.explainer.explain(
                        model=self.agent.model,
                        X=X,
                        instance_id=instance_id
                    )

                    response = {
                        "agent": str(self.agent.jid),
                        "prediction": y_pred,
                        "confidence": confidence,
                        "explanation": explanation
                    }

                    reply = Message(to=str(msg.sender))
                    reply.set_metadata("performative", "inform")
                    reply.body = json.dumps(response)
                    await self.send(reply)

                elif action == "feedback":
                    # Feedback externo para ajustar hiperparámetros
                    feedback_metrics = content.get("metrics")
                    if feedback_metrics:
                        self.agent.adjust_from_feedback(feedback_metrics)
                        print(f"[{self.agent.jid}] Ajuste realizado según feedback: {feedback_metrics}")

            except Exception as e:
                error_msg = Message(to=str(msg.sender))
                error_msg.set_metadata("performative", "failure")
                error_msg.body = json.dumps({"error": str(e)})
                await self.send(error_msg)

    # ==================== Métodos de ajuste ====================
    def auto_tune_best(self, X_train, y_train, X_test, y_test):
        """
        Método simple de autoajuste:
        se puede reemplazar por grid search, optuna, etc.
        """
        best_acc = 0
        best_model = None

        # Ejemplo: variar un parámetro si es XGB o RF
        for param_change in [0.8, 1.0, 1.2]:
            if hasattr(self.model, "set_params"):
                try:
                    # Suponiendo que tenga learning_rate o n_estimators
                    params = self.model.get_params()
                    if "learning_rate" in params:
                        self.model.set_params(learning_rate=params["learning_rate"] * param_change)
                    elif "n_estimators" in params:
                        self.model.set_params(n_estimators=int(params["n_estimators"] * param_change))

                    self.model.fit(X_train, y_train)
                    acc = accuracy_score(y_test, self.model.predict(X_test))
                    if acc > best_acc:
                        best_acc = acc
                        best_model = self.model

                except Exception:
                    continue

        if best_model is not None:
            self.model = best_model
            self.metrics_history.append({"accuracy": best_acc})
            print(f"[{self.jid}] Autoajuste completado. Mejor accuracy: {best_acc:.3f}")

    def adjust_from_feedback(self, feedback_metrics):
        """
        Ajusta el modelo basado en métricas externas proporcionadas como feedback.
        Ejemplo: ajustar learning_rate, profundidad máxima, etc.
        """
        # Aquí se pueden implementar heurísticas de ajuste según feedback
        if hasattr(self.model, "set_params"):
            params = self.model.get_params()
            if "learning_rate" in params and "accuracy" in feedback_metrics:
                # Aumentar ligeramente si accuracy < deseada
                if feedback_metrics["accuracy"] > self.metrics_history[-1]["accuracy"]:
                    params["learning_rate"] *= 1.05
                else:
                    params["learning_rate"] *= 0.95
                self.model.set_params(**params)
                print(f"[{self.jid}] Parámetro learning_rate ajustado según feedback")
