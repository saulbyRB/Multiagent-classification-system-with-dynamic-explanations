import numpy as np


class HybridEvaluator:
    """
    Evaluador híbrido multi-métrica y explicación-aware
    """

    def __init__(
        self,
        w_pred=0.35,
        w_conf=0.15,
        w_exp=0.25,
        w_perf=0.25
    ):
        self.w_pred = w_pred
        self.w_conf = w_conf
        self.w_exp  = w_exp
        self.w_perf = w_perf

    def evaluate(self, responses):
        preds = [r["prediction"] for r in responses]
        confs = [r.get("confidence", 0.0) for r in responses]
        exps  = [self._aggregate_explanations(r.get("explanations", [])) for r in responses]
        perfs = [r.get("metrics", {}) for r in responses]

        majority = self._majority_vote(preds)

        S_pred = np.array([1.0 if p == majority else 0.0 for p in preds])
        S_conf = np.array(self._normalize(confs))
        S_exp  = np.array(self._explanation_scores(exps))
        S_perf = np.array(self._performance_scores(perfs))

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
                "exp": S_exp.tolist(),
                "perf": S_perf.tolist()
            }
        }

    # ---------- helpers ----------

    def _majority_vote(self, preds):
        return max(set(preds), key=preds.count)

    def _normalize(self, values, eps=1e-8):
        v = np.array(values, dtype=float)
        return ((v - v.min()) / (v.max() - v.min() + eps)).tolist()

    def _aggregate_explanations(self, explanations):
        if not explanations:
            return None
        vectors = [np.array(exp["details"]["values"]) for exp in explanations]
        return np.mean(vectors, axis=0)

    def _explanation_scores(self, vectors):
        valid = [v for v in vectors if v is not None]
        if not valid:
            return [0.0] * len(vectors)

        center = np.mean(valid, axis=0)
        return [
            self._cosine(v, center) if v is not None else 0.0
            for v in vectors
        ]

    def _performance_scores(self, metrics):
        """
        Combina métricas clásicas (acc, f1, recall, etc.)
        """
        scores = []
        for m in metrics:
            if not m:
                scores.append(0.0)
                continue

            score = 0.0
            score += 0.4 * m.get("f1", 0.0)
            score += 0.3 * m.get("accuracy", 0.0)
            score += 0.2 * m.get("precision", 0.0)
            score += 0.1 * m.get("recall", 0.0)

            scores.append(score)

        return self._normalize(scores)

    def _cosine(self, a, b, eps=1e-8):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
