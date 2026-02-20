# classifier_agent.py
import asyncio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from agents.message import Message
from visualization.logs import log


class ClassifierAgent:

    def __init__(
        self,
        agent_id,
        model,
        explainers,
        dataset_id,
        registry,
        test_size=0.2
    ):
        self.id = agent_id
        self.model = model
        self.explainers = explainers
        self.dataset_id = dataset_id
        self.registry = registry
        self.test_size = test_size

        self.inbox = asyncio.Queue()
        self.current_iteration = 0
        self.metrics_history = []
        self.explanations_history = []

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.running = True

    # ---------- setup ----------

    async def setup(self):
        print(f"[{self.id}] Setup iniciado")

        X, y, meta = self.registry.load(self.dataset_id)

        for explainer in self.explainers:
            if hasattr(explainer, "set_background"):
                explainer.set_background(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size
        )

        log("Setup iniciado (train + eval inicial)", self.id)
        self._fit_and_evaluate(initial=True)
        log("Setup completado", self.id)

    # ---------- main loop ----------

    async def run(self, queues):
        while self.running:
            msg: Message = await self.inbox.get()
            await self.handle_message(msg, queues)

    # ---------- message handling ----------

    async def handle_message(self, msg: Message, queues):
        action = msg.body["action"]

        if action == "classify":
            await self._handle_classify(msg, queues)

        elif action == "feedback":
            self.adjust_from_feedback(msg.body)

        elif action == "shutdown":
            log("Shutdown recibido", self.id)
            self.running = False

    # ---------- classify ----------

    async def _handle_classify(self, msg, queues):
        iteration = msg.body["iteration"]
        instance = msg.body["instance"]

        log("Clasificando instancia", self.id)

        y_pred = int(self.model.predict(instance)[0])

        confidence = self._estimate_confidence(instance)

        explanations = self._generate_explanations(instance)
    
        explanations = self._generate_explanations(instance)

        if explanations:
            log("Calculando explicaciones", self.id)

            # ⬇️ NUEVO: guardar vector explicativo
            # asumimos una explicación principal (ej. SHAP)
            main_exp = explanations[0]

            if "details" in main_exp and "values" in main_exp["details"]:
                values = np.array(main_exp["details"]["values"])
                self.explanation_history.append(values)

        response = {
            "agent_id": self.id,
            "iteration": iteration,
            "prediction": y_pred,
            "confidence": confidence,
            "metrics": self.metrics_history[-1],
            "explanations": explanations
        }
        
        log("Clasificación completada", self.id)

        await queues["aggregator"].put(
            Message(sender=self.id, body=response)
        )

    # ---------- feedback ----------

    def adjust_from_feedback(self, feedback):
        """
        feedback contiene:
        - iteration
        - strategy
        """
        self.current_iteration = feedback["iteration"]
        strategy = feedback.get("strategy")

        log(
            f"Ajustando modelo | iter={self.current_iteration} | strategy={strategy}",
            self.id
        )

        if strategy in {"adjust", "soft_adjust", "force_adjust"}:
            self.model.fit(self.X_train, self.y_train)

        self._fit_and_evaluate(initial=False)

        print(
            f"[{self.id}] Iter {self.current_iteration} | "
            f"Estrategia={strategy} | "
            f"F1={self.metrics_history[-1]['f1']:.3f}"
        )

    # ---------- helpers ----------

    def _fit_and_evaluate(self, initial=False):
        # Entrenar si es inicial o si el modelo aún no está entrenado
        if initial or not self.model.is_trained:
            self.model.fit(self.X_train, self.y_train)

        log(f"Entrenando modelo (iter={self.current_iteration}, initial={initial})", self.id)

        y_pred = self.model.predict(self.X_test)

        metrics = {
            "iteration": self.current_iteration,
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred, average="weighted"),
            "precision": precision_score(self.y_test, y_pred, average="weighted"),
            "recall": recall_score(self.y_test, y_pred, average="weighted")
        }
        log(f"Evaluación completada | acc={metrics['accuracy']:.3f}", self.id)

        self.metrics_history.append(metrics)


    def _estimate_confidence(self, instance):
        """
        Estimación genérica de confianza
        """
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(instance)[0]
            return float(np.max(proba))
        return 0.5

    def _generate_explanations(self, instance):
        explanations = []

        for exp in self.explainers:
            e = exp.explain(self.model, instance, 0)

            # validación mínima esperada por el evaluador
            if "details" in e and "values" in e["details"]:
                explanations.append(e)

        return explanations
