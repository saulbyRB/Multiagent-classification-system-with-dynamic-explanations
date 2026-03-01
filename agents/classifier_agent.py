import asyncio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from agents.message import Message
from visualization.logs import log
from models.torch_model import TorchModel


class ClassifierAgent:

    def __init__(self, agent_id, model, explainers,
                 dataset_id, registry, test_size=0.2):
        self.id          = agent_id
        self.model       = model
        self.explainers  = explainers
        self.dataset_id  = dataset_id
        self.registry    = registry
        self.test_size   = test_size

        self.inbox        = asyncio.Queue()
        self.current_iteration = 0
        self.metrics_history   = []
        self.explanation_history = []

        self.X_train = self.y_train = None
        self.X_test  = self.y_test  = None
        self.running = True

    async def setup(self):
        print(f"[{self.id}] Setup iniciado")
        X, y, meta = self.registry.load(self.dataset_id)

        for explainer in self.explainers:
            if hasattr(explainer, "set_background"):
                explainer.set_background(X)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=self.test_size)

        log("Setup iniciado (train + eval inicial)", self.id)
        self._fit_and_evaluate(initial=True)
        log("Setup completado", self.id)

    async def run(self, queues):
        while self.running:
            msg: Message = await self.inbox.get()
            await self.handle_message(msg, queues)

    async def handle_message(self, msg: Message, queues):
        action = msg.body["action"]
        if action == "classify":
            await self._handle_classify(msg, queues)
        elif action == "feedback":
            self.adjust_from_feedback(msg.body)
        elif action == "shutdown":
            log("Shutdown recibido", self.id)
            self.running = False

    async def _handle_classify(self, msg, queues):
        instance  = msg.body["instance"]
        iteration = msg.body["iteration"]

        log("Clasificando instancia", self.id)
        y_pred       = int(self.model.predict(instance)[0])
        confidence   = self._estimate_confidence(instance)
        explanations = self._generate_explanations(instance)

        if not hasattr(self, "explanation_history_by_type"):
            self.explanation_history_by_type = {}

        for e in explanations:
            name = e.get("explainer", "unknown")
            if "details" in e and "values" in e["details"]:
                v = np.array(e["details"]["values"])
                if name not in self.explanation_history_by_type:
                    self.explanation_history_by_type[name] = []
                self.explanation_history_by_type[name].append(v)

        if explanations:
            vectors = [
                np.array(e["details"]["values"])
                for e in explanations
                if "details" in e and "values" in e["details"]
            ]
            if vectors:
                self.explanation_history.append(np.mean(vectors, axis=0))

        response = {
            "agent_id":    self.id,
            "iteration":   iteration,
            "prediction":  y_pred,
            "confidence":  confidence,
            "metrics":     self.metrics_history[-1],
            "explanations": explanations,
            "exp_history": getattr(self, "explanation_history", []),
            "exp_history_by_type": getattr(
                self, "explanation_history_by_type", {}),
            "instance":   instance,
            "model_ref":  self.model
        }

        log("Clasificación completada", self.id)
        await queues["aggregator"].put(
            Message(sender=self.id, body=response)
        )

    def adjust_from_feedback(self, feedback):
        self.current_iteration = feedback["iteration"]
        strategy       = feedback.get("strategy")
        evaluation     = feedback.get("evaluation", {})
        trend          = feedback.get("trend", 0.0)
        global_trend   = feedback.get("global_trend", 0.0)
        group_signals  = feedback.get("group_signals", {})

        group_pressure          = group_signals.get("group_pressure", 0.0)
        all_peers_struggling    = group_signals.get("all_peers_struggling", False)
        relative_position       = group_signals.get("relative_position", 0.0)

        log(
            f"Ajustando modelo | iter={self.current_iteration} | "
            f"strategy={strategy} | trend={trend:+.3f} | "
            f"pos_relativa={relative_position:+.3f}",
            self.id
        )

        if hasattr(self.model, "adjust_from_feedback"):
            signals = {
                "strategy":             strategy,
                "reward":               evaluation.get("reward", 0.5),
                "trend":                trend,
                "global_trend":         global_trend,
                # Señales de grupo
                "group_pressure":       group_pressure,
                "all_peers_struggling": all_peers_struggling,
                "relative_position":    relative_position,
                "peer_scores":          group_signals.get("peer_scores", []),
            }
            if isinstance(self.model, TorchModel):
                self.model.adjust_from_feedback(
                    signals, self.X_train, self.y_train)
            else:
                self.model.adjust_from_feedback(
                    signals, self.X_train, self.y_train,
                    self.X_test, self.y_test)
        elif strategy in {"adjust", "force_adjust"}:
            idx = np.random.permutation(len(self.X_train))
            self.model.fit(self.X_train[idx], self.y_train[idx])

        if strategy in {"adjust", "force_adjust", "soft_adjust"}:
            for exp in self.explainers:
                if hasattr(exp, "invalidate"):
                    exp.invalidate()
                elif hasattr(exp, "_explainer"):
                    exp._explainer = None

        self._fit_and_evaluate(initial=False)

    def _fit_and_evaluate(self, initial=False):
        if initial or not self.model.is_trained:
            log(f"Entrenando modelo (iter={self.current_iteration})", self.id)
            self.model.fit(self.X_train, self.y_train)
            for exp in self.explainers:
                if hasattr(exp, "_explainer"):
                    exp._explainer = None

        y_pred  = self.model.predict(self.X_test)
        metrics = {
            "iteration": self.current_iteration,
            "accuracy":  accuracy_score(self.y_test, y_pred),
            "f1":        f1_score(self.y_test, y_pred, average="weighted"),
            "precision": precision_score(
                self.y_test, y_pred, average="weighted"),
            "recall":    recall_score(
                self.y_test, y_pred, average="weighted")
        }
        log(f"Evaluación completada | acc={metrics['accuracy']:.3f}", self.id)
        self.metrics_history.append(metrics)

    def _estimate_confidence(self, instance):
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(instance)[0]
            return float(np.max(proba))
        return 0.5

    def _generate_explanations(self, instance):
        explanations = []
        for exp in self.explainers:
            e = exp.explain(self.model, instance, 0)
            if "details" in e and "values" in e["details"]:
                explanations.append(e)
        return explanations