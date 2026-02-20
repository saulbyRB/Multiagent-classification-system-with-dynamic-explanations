import numpy as np


class HybridEvaluator:
    """
    Evaluador híbrido multi-métrica y explicación-aware (profundo).
    """

    def __init__(
        self,
        w_pred=0.30,
        w_conf=0.15,
        w_exp=0.30,
        w_perf=0.25
    ):
        self.w_pred = w_pred
        self.w_conf = w_conf
        self.w_exp  = w_exp
        self.w_perf = w_perf

        # pesos internos de explicación
        self.w_cons = 0.35
        self.w_stab = 0.25
        self.w_qual = 0.20
        self.w_fid  = 0.20

    # =========================================================
    # MAIN
    # =========================================================

    def evaluate(self, responses):
        preds = [r["prediction"] for r in responses]
        confs = [r.get("confidence", 0.0) for r in responses]
        perfs = [r.get("metrics", {}) for r in responses]

        expls = [
            self._aggregate_explanations(r.get("explanations", []))
            for r in responses
        ]

        histories = [
            r.get("exp_history", [])
            for r in responses
        ]

        majority = self._majority_vote(preds)

        S_pred = np.array([1.0 if p == majority else 0.0 for p in preds])
        S_conf = np.array(self._normalize(confs))
        S_perf = np.array(self._performance_scores(perfs))

        S_exp, exp_components = self._explanation_score(expls, histories)

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
                "exp_detail": exp_components
            }
        }

    # =========================================================
    # EXPLANATION SCORING
    # =========================================================

    def _explanation_score(self, vectors, histories):
        consensus = self._consensus_score(vectors)
        stability = self._stability_score(histories)
        quality   = self._quality_score(vectors)
        fidelity  = self._fidelity_score(vectors)

        S = (
            self.w_cons * consensus +
            self.w_stab * stability +
            self.w_qual * quality +
            self.w_fid  * fidelity
        )

        return self._normalize(S), {
            "consensus": consensus.tolist(),
            "stability": stability.tolist(),
            "quality": quality.tolist(),
            "fidelity": fidelity.tolist()
        }

    # ---------------------------------------------------------

    def _consensus_score(self, vectors):
        valid = [v for v in vectors if v is not None]
        if not valid:
            return np.zeros(len(vectors))

        center = np.mean(valid, axis=0)
        return np.array([
            self._cosine(v, center) if v is not None else 0.0
            for v in vectors
        ])

    def _stability_score(self, histories):
        scores = []
        for h in histories:
            if len(h) < 2:
                scores.append(0.5)
            else:
                sims = [
                    self._cosine(h[i-1], h[i])
                    for i in range(1, len(h))
                ]
                scores.append(float(np.mean(sims)))
        return np.array(scores)

    def _quality_score(self, vectors):
        scores = []
        for v in vectors:
            if v is None:
                scores.append(0.0)
                continue

            sparsity = np.mean(np.abs(v) > 1e-3)
            energy   = np.linalg.norm(v)

            scores.append(
                0.6 * sparsity +
                0.4 * np.tanh(energy)
            )
        return np.array(scores)

    def _fidelity_score(self, vectors):
        scores = []
        for v in vectors:
            if v is None:
                scores.append(0.0)
                continue

            alignment = np.mean(np.abs(v))
            scores.append(np.tanh(alignment))
        return np.array(scores)

    # =========================================================
    # AUX
    # =========================================================

    def _majority_vote(self, preds):
        return max(set(preds), key=preds.count)

    def _normalize(self, values, eps=1e-8):
        v = np.array(values, dtype=float)
        return (v - v.min()) / (v.max() - v.min() + eps)

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

    def _cosine(self, a, b, eps=1e-8):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))