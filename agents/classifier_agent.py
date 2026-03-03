import asyncio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from agents.message import Message
from visualization.logs import log
from models.torch_model import TorchModel


# ── Cooldown config ────────────────────────────────────────────────────────────
# Peso que suma cada strategy al contador de cooldown.
# force_adjust nunca bloquea; soft_adjust contribuye a la mitad.
_COOLDOWN_WEIGHT = {
    "force_adjust": 0,
    "adjust":       1,
    "soft_adjust":  0.5,
}

# Umbral de cooldown según clase de modelo.
# Las redes necesitan más iteraciones para converger.
_COOLDOWN_THRESHOLD = {
    "TorchModel":               8,
    "GradientBoostingModel":    5,
    "RandomForest":             5,
    "default":                  3,
}

# Cada cuántas iteraciones se "perdona" 1 punto del contador,
# aunque no haya habido mejora real. Evita bloqueos permanentes.
_COOLDOWN_DECAY_EVERY = 10


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
        self._iters_since_adjust  = 0
        self._consecutive_adjusts = 0.0   # ahora es float para pesos fraccionarios

    # ── Umbral de cooldown dinámico según tipo de modelo ──────────────────
    @property
    def _cooldown_threshold(self):
        class_name = self.model.__class__.__name__
        return _COOLDOWN_THRESHOLD.get(class_name, _COOLDOWN_THRESHOLD["default"])

    async def setup(self):
        print(f"[{self.id}] Setup iniciado")
        X, y, meta = self.registry.load(self.dataset_id)

        for explainer in self.explainers:
            if hasattr(explainer, "set_background"):
                explainer.set_background(X)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=self.test_size, random_state=42)

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

        # Invalidar explainers en cada clasificación para evitar caché estático
        for exp in self.explainers:
            if hasattr(exp, "invalidate"):
                exp.invalidate()
            elif hasattr(exp, "_explainer"):
                exp._explainer = None

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

        self._iters_since_adjust += 1

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

        response["iters_since_adjust"] = self._iters_since_adjust

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

        # ── Cooldown ───────────────────────────────────────────────────────
        if strategy in _COOLDOWN_WEIGHT:
            weight    = _COOLDOWN_WEIGHT[strategy]
            threshold = self._cooldown_threshold

            # force_adjust (weight=0) nunca activa ni respeta el cooldown
            if weight == 0:
                log("force_adjust → ignorando cooldown", self.id)
            else:
                # Decay periódico: cada N iters sin ajuste se perdona 1 punto
                if (self.current_iteration > 0
                        and self.current_iteration % _COOLDOWN_DECAY_EVERY == 0):
                    old = self._consecutive_adjusts
                    self._consecutive_adjusts = max(0.0, self._consecutive_adjusts - 1.0)
                    if self._consecutive_adjusts < old:
                        log(
                            f"Cooldown decay aplicado: "
                            f"{old:.1f} → {self._consecutive_adjusts:.1f}",
                            self.id
                        )

                # Actualizar contador según mejora de accuracy
                if len(self.metrics_history) >= 2:
                    delta = (self.metrics_history[-1]["accuracy"]
                             - self.metrics_history[-2]["accuracy"])
                    if delta > 0.001:
                        # Mejora real → reset completo
                        self._consecutive_adjusts = 0.0
                    else:
                        # Sin mejora → sumar peso fraccionario según strategy
                        self._consecutive_adjusts += weight
                else:
                    self._consecutive_adjusts = 0.0

                # Verificar si debemos saltar el re-entrenamiento
                if self._consecutive_adjusts >= threshold:
                    log(
                        f"Cooldown activado "
                        f"({self._consecutive_adjusts:.1f}/{threshold} "
                        f"[{self.model.__class__.__name__}]) "
                        f"→ skip re-entrenamiento",
                        self.id
                    )
                    self._iters_since_adjust = 0
                    return

        # ── Ajuste del modelo ──────────────────────────────────────────────
        if hasattr(self.model, "adjust_from_feedback"):
            signals = {
                "strategy":             strategy,
                "reward":               evaluation.get("reward", 0.5),
                "trend":                trend,
                "global_trend":         global_trend,
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

        # ── Invalidar explainers tras ajuste ──────────────────────────────
        if strategy in {"adjust", "force_adjust", "soft_adjust"}:
            for exp in self.explainers:
                if hasattr(exp, "invalidate"):
                    exp.invalidate()
                elif hasattr(exp, "_explainer"):
                    exp._explainer = None

        self._iters_since_adjust = 0
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

        for explainer in self.explainers:
            try:
                result = explainer.explain(self.model, instance)
                if result:
                    explanations.append(result)
            except Exception as ex:
                log(f"Error en explainer {explainer}: {ex}", self.id)

        # Debug temporal
        for e in explanations:
            if "details" in e and "values" in e["details"]:
                v = np.array(e["details"]["values"])
                top3 = np.argsort(np.abs(v))[::-1][:3]
                feature_names = e["details"].get("feature_names", [])
                top3_names = [feature_names[i] for i in top3] if feature_names else top3.tolist()
                print(f"[{self.id}] {e.get('explainer')}: top3={top3_names} | top3_idx={top3.tolist()}")
                print(f"[{self.id}] {e.get('explainer')}: valores={np.round(v[top3], 4).tolist()}")

        return explanations