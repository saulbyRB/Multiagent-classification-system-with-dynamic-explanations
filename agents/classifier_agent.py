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

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

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

        self._fit_and_evaluate(initial=True)

    # ---------- main loop ----------

    async def run(self, queues):
        while True:
            msg: Message = await self.inbox.get()
            await self.handle_message(msg, queues)

    # ---------- message handling ----------

    async def handle_message(self, msg: Message, queues):
        action = msg.body["action"]

        if action == "classify":
            await self._handle_classify(msg, queues)

        elif action == "feedback":
            self.adjust_from_feedback(msg.body)

    # ---------- classify ----------

    async def _handle_classify(self, msg, queues):
        iteration = msg.body["iteration"]
        instance = msg.body["instance"]

        y_pred = int(self.model.predict(instance)[0])

        confidence = self._estimate_confidence(instance)

        explanations = self._generate_explanations(instance)

        response = {
            "agent_id": self.id,
            "iteration": iteration,
            "prediction": y_pred,
            "confidence": confidence,
            "metrics": self.metrics_history[-1],
            "explanations": explanations
        }

        await queues["aggregator"].put(
            Message(sender=self.id, body=response)
        )

    # ---------- feedback ----------

    def adjust_from_feedback(self, feedback):
        """
        feedback contiene:
        - strategy
        - evaluation (scores, componentes, etc.)
        """
        self.current_iteration += 1

        strategy = feedback.get("strategy")

        # 🔴 Ajuste aún simulado (se implementará después)
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
        if not initial:
            self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)

        metrics = {
            "iteration": self.current_iteration,
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred, average="weighted"),
            "precision": precision_score(self.y_test, y_pred, average="weighted"),
            "recall": recall_score(self.y_test, y_pred, average="weighted")
        }

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
