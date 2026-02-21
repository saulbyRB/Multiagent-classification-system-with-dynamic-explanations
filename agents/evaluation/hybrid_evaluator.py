import numpy as np


class HybridEvaluator:
    """
    Evaluador híbrido multi-métrica y explanation-aware.
    """

    def __init__(
        self,
        w_pred=0.3,
        w_conf=0.15,
        w_exp=0.3,
        w_perf=0.25
    ):
        self.w_pred = w_pred
        self.w_conf = w_conf
        self.w_exp  = w_exp
        self.w_perf = w_perf

    # ======================================================
    # Main
    # ======================================================

    def evaluate(self, responses):
        preds = [r["prediction"] for r in responses]
        confs = [r.get("confidence", 0.0) for r in responses]
        perfs = [r.get("metrics", {}) for r in responses]

        # --- explicaciones actuales ---
        expl_vectors = [
            self._aggregate_explanations(r.get("explanations", []))
            for r in responses
        ]

        # --- historial explicativo ---
        histories = [
            r.get("exp_history", [])
            for r in responses
        ]

        majority = self._majority_vote(preds)

        S_pred = np.array([1.0 if p == majority else 0.0 for p in preds])
        S_conf = np.array(self._normalize(confs))
        S_perf = np.array(self._performance_scores(perfs))

        # ⬇️ métricas explicativas detalladas
        consensus, stability, S_exp = self._explanation_quality(
            expl_vectors, histories
        )

        scores = (
            self.w_pred * S_pred +
            self.w_conf * S_conf +
            self.w_exp  * S_exp +
            self.w_perf * S_perf
        )

        return {
            "global_score": float(scores.mean()),
            "majority_prediction": majority,
            "scores": scores.tolist(),
            "components": {
                "pred": S_pred.tolist(),
                "conf": S_conf.tolist(),
                "perf": S_perf.tolist(),
                "exp": S_exp.tolist(),
                # 🔹 AQUÍ lo que querías
                "exp_detail": {
                    "consensus": consensus,
                    "stability": stability,
                    "quality": S_exp.tolist()
                }
            }
        }

    # ======================================================
    # Explicaciones
    # ======================================================

    def _aggregate_explanations(self, explanations):
        if not explanations:
            return None

        vectors = [
            np.asarray(exp["details"]["values"], dtype=float)
            for exp in explanations
            if "details" in exp and "values" in exp["details"]
        ]

        if not vectors:
            return None

        return np.mean(vectors, axis=0)

    def _explanation_quality(self, vectors, histories):
        """
        Devuelve:
        - consensus: consenso inter-agente
        - stability: estabilidad temporal
        - quality: combinación de ambas
        """
        valid = [v for v in vectors if v is not None]

        if not valid:
            n = len(vectors)
            return (
                [0.0] * n,
                [0.5] * n,
                np.zeros(n)
            )

        # -------- consenso inter-agente --------
        center = np.mean(valid, axis=0)
        consensus = [
            self._cosine(v, center) if v is not None else 0.0
            for v in vectors
        ]

        # -------- estabilidad temporal --------
        stability = []
        for v, hist in zip(vectors, histories):
            if v is None or len(hist) == 0:
                stability.append(0.5)
            else:
                prev = np.mean(hist, axis=0)
                stability.append(self._cosine(v, prev))

        # -------- calidad final --------
        quality = np.array([
            0.6 * c + 0.4 * s
            for c, s in zip(consensus, stability)
        ])

        return consensus, stability, quality

    # ======================================================
    # Métricas clásicas
    # ======================================================

    def _performance_scores(self, metrics):
        scores = []
        for m in metrics:
            if not m:
                scores.append(0.0)
                continue

            scores.append(
                0.4 * m.get("f1", 0.0) +
                0.3 * m.get("accuracy", 0.0) +
                0.2 * m.get("precision", 0.0) +
                0.1 * m.get("recall", 0.0)
            )

        return self._normalize(scores)

    # ======================================================
    # Utils
    # ======================================================

    def _majority_vote(self, preds):
        return max(set(preds), key=preds.count)

    def _normalize(self, values, eps=1e-8):
        v = np.array(values, dtype=float)
        if v.max() - v.min() < eps:
            return [0.5] * len(v)
        return ((v - v.min()) / (v.max() - v.min() + eps)).tolist()

    def _cosine(self, a, b, eps=1e-8):
        return float(
            np.dot(a, b) /
            (np.linalg.norm(a) * np.linalg.norm(b) + eps)
        )