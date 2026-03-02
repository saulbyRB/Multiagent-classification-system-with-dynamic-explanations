import numpy as np


class HybridEvaluator:
    """
    Evaluador híbrido multi-métrica y explanation-aware.

    Métricas explicativas calculadas POR SEPARADO para cada explainer:
    - consensus:  similitud coseno inter-agente
    - stability:  estabilidad temporal usando historial por tipo de explainer
    - fidelity:   sensibilidad de las top-features vs features aleatorias (escala-aware)
    - agreement:  acuerdo entre explainers del mismo agente (SHAP vs LIME)
    - quality:    combinación ponderada de las cuatro anteriores
    """

    def __init__(
        self,
        w_pred=0.3,
        w_conf=0.15,
        w_exp=0.3,
        w_perf=0.25,
        background_data=None
    ):
        self.w_pred = w_pred
        self.w_conf = w_conf
        self.w_exp  = w_exp
        self.w_perf = w_perf

        # Acumulador de instancias para estimar std por feature
        if background_data is not None:
            std = np.std(background_data, axis=0)
            self._feature_std = np.where(std > 1e-6, std, 1.0)
            self._background = np.asarray(background_data, dtype=float)
        else:
            self._feature_std = None
            self._background  = None

    # ======================================================
    # Main
    # ======================================================

    def evaluate(self, responses):
        preds = [r["prediction"] for r in responses]
        confs = [r.get("confidence", 0.0) for r in responses]
        perfs = [r.get("metrics", {}) for r in responses]

        explanations_by_type = self._split_by_explainer(responses)
        histories_by_type    = [r.get("exp_history_by_type", {}) for r in responses]

        instances = [r.get("instance", None) for r in responses]
        models    = [r.get("model_ref", None) for r in responses]

        majority = self._majority_vote(preds)

        S_pred = np.array([1.0 if p == majority else 0.0 for p in preds])
        S_conf = np.array(self._normalize(confs))
        S_perf = np.array(self._performance_scores(perfs))

        exp_result = self._explanation_quality_multi(
            explanations_by_type, histories_by_type, instances, models, preds
        )

        S_exp = exp_result["quality"]

        scores = (
            self.w_pred * S_pred +
            self.w_conf * S_conf +
            self.w_exp  * S_exp +
            self.w_perf * S_perf
        )

        # En evaluate(), después de calcular S_perf, añadir:
        S_acc = np.array([m.get("accuracy", 0.0) for m in perfs])

        # Y en el return, dentro de components:
        return {
            "global_score":        float(scores.mean()),
            "majority_prediction": majority,
            "scores":              scores.tolist(),
            "components": {
                "pred": S_pred.tolist(),
                "conf": S_conf.tolist(),
                "perf": S_perf.tolist(),
                "exp":  S_exp.tolist(),
                "acc":  S_acc.tolist(),    # ← añadir esto
                "exp_detail": {
                    "consensus":     exp_result["consensus"],
                    "stability":     exp_result["stability"],
                    "fidelity":      exp_result["fidelity"],
                    "agreement":     exp_result["agreement"],
                    "quality":       S_exp.tolist(),
                    "per_explainer": exp_result["per_explainer"]
                }
            }
        }

    # ======================================================
    # Feature std estimator
    # ======================================================

    def _update_feature_std(self, instances):
        """
        Acumula instancias vistas y recalcula std por feature.
        Con ≥5 instancias la estimación ya es útil.
        """
        for inst in instances:
            if inst is not None:
                self._instance_buffer.append(
                    np.asarray(inst, dtype=float).flatten()
                )

        if len(self._instance_buffer) >= 2:
            buf = np.stack(self._instance_buffer, axis=0)
            std = np.std(buf, axis=0)
            # Evitar std=0 (feature constante): fallback a 1.0
            self._feature_std = np.where(std > 1e-6, std, 1.0)

    def _get_delta(self, feature_idx, n_features):
        """
        Delta de perturbación = 0.5 stds de la feature.
        Si aún no hay estimación suficiente usa 0.5 como fallback.
        """
        if self._feature_std is not None and feature_idx < len(self._feature_std):
            return float(self._feature_std[feature_idx] * 0.5)
        return 0.5

    # ======================================================
    # Split por tipo de explainer
    # ======================================================

    def _split_by_explainer(self, responses):
        explainer_names = set()
        for r in responses:
            for exp in r.get("explanations", []):
                explainer_names.add(exp.get("explainer", "unknown"))

        result = {}
        for name in explainer_names:
            vectors = []
            for r in responses:
                match = next(
                    (e for e in r.get("explanations", [])
                     if e.get("explainer") == name
                     and "details" in e
                     and "values" in e["details"]),
                    None
                )
                if match:
                    v = np.asarray(match["details"]["values"], dtype=float)
                    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    vectors.append(v if np.linalg.norm(v) > 1e-8 else None)
                else:
                    vectors.append(None)
            result[name] = vectors

        return result

    # ======================================================
    # Calidad explicativa multi-explainer
    # ======================================================

    def _explanation_quality_multi(
        self, explanations_by_type, histories_by_type, instances, models, preds
    ):
        n = len(histories_by_type)
        explainer_names = list(explanations_by_type.keys())

        per_explainer = {}
        all_consensus = np.zeros(n)
        all_stability = np.zeros(n)
        all_fidelity  = np.zeros(n)
        valid_count   = np.zeros(n)

        for name in explainer_names:
            vectors = explanations_by_type[name]
            valid   = [v for v in vectors if v is not None]

            if not valid:
                per_explainer[name] = {
                    "consensus": [0.0] * n,
                    "stability": [0.5] * n,
                    "fidelity":  [0.5] * n,
                }
                continue

            center    = np.mean(valid, axis=0)
            consensus = [
                self._cosine(v, center) if v is not None else 0.0
                for v in vectors
            ]

            stability = []
            for v, hist_by_type in zip(vectors, histories_by_type):
                hist = hist_by_type.get(name, [])
                if v is None or len(hist) == 0:
                    stability.append(0.5)
                else:
                    prev = np.mean(hist, axis=0)
                    stability.append(self._cosine(v, prev))

            fidelity = [
                self._compute_fidelity(v, inst, model, pred)
                for v, inst, model, pred in zip(vectors, instances, models, preds)
            ]

            per_explainer[name] = {
                "consensus": consensus,
                "stability": stability,
                "fidelity":  fidelity,
            }

            for i, v in enumerate(vectors):
                if v is not None:
                    all_consensus[i] += consensus[i]
                    all_stability[i] += stability[i]
                    all_fidelity[i]  += fidelity[i]
                    valid_count[i]   += 1

        safe_count     = np.where(valid_count > 0, valid_count, 1)
        mean_consensus = (all_consensus / safe_count).tolist()
        mean_stability = (all_stability / safe_count).tolist()
        mean_fidelity  = (all_fidelity  / safe_count).tolist()

        agreement = self._compute_agreement(explanations_by_type, n)

        quality = np.array([
            0.30 * c + 0.25 * s + 0.30 * f + 0.15 * a
            for c, s, f, a in zip(
                mean_consensus, mean_stability, mean_fidelity, agreement
            )
        ])

        return {
            "consensus":     mean_consensus,
            "stability":     mean_stability,
            "fidelity":      mean_fidelity,
            "agreement":     agreement,
            "quality":       quality,
            "per_explainer": per_explainer
        }

    # ======================================================
    # Agreement inter-explainer
    # ======================================================

    def _compute_agreement(self, explanations_by_type, n):
        explainer_names = list(explanations_by_type.keys())

        if len(explainer_names) < 2:
            return [1.0] * n

        agreement = []
        for i in range(n):
            top_features = []
            for name in explainer_names:
                v = explanations_by_type[name][i]
                if v is not None and np.linalg.norm(v) > 1e-8:
                    top_features.append(int(np.argmax(np.abs(v))))

            if len(top_features) < 2:
                agreement.append(0.5)
                continue

            pairs = matches = 0
            for a in range(len(top_features)):
                for b in range(a + 1, len(top_features)):
                    pairs += 1
                    if top_features[a] == top_features[b]:
                        matches += 1

            agreement.append(matches / pairs)

        return agreement

    # ======================================================
    # Fidelity — perturbación escala-aware con baseline
    # ======================================================

    def _compute_fidelity(self, expl_vector, instance, model, original_pred):
        if expl_vector is None or instance is None or model is None:
            return 0.5

        try:
            v = np.asarray(expl_vector, dtype=float)
            n_features = v.shape[0]

            k     = max(1, int(0.3 * n_features))
            top_k = set(np.argsort(np.abs(v))[::-1][:k].tolist())
            rest  = [i for i in range(n_features) if i not in top_k]

            if not rest:
                return 0.5

            # Usar instancias del background como sujetos de perturbación
            # → variedad garantizada, algunas cerca de la frontera
            if self._background is not None and len(self._background) >= 5:
                idxs = np.random.choice(len(self._background), size=20, replace=False)
                probe_set = self._background[idxs]
            else:
                probe_set = np.asarray(instance, dtype=float).reshape(1, -1)

            def mean_sensitivity(feature_indices):
                effects = []
                for x_probe in probe_set:
                    x = x_probe.reshape(1, -1)
                    y0 = int(model.predict(x)[0])
                    for fi in feature_indices:
                        delta = self._get_delta(fi, n_features)
                        for sign in (+1, -1):
                            xi = x.copy()
                            xi[0, fi] += sign * delta
                            yi = int(model.predict(xi)[0])
                            if yi != y0:
                                effects.append(1.0)
                                break
                        else:
                            effects.append(0.0)
                return float(np.mean(effects)) if effects else 0.0

            top_score  = mean_sensitivity(list(top_k))
            base_idxs  = np.random.choice(rest, size=min(k, len(rest)), replace=False)
            base_score = mean_sensitivity(base_idxs.tolist()) + 0.05

            ratio    = top_score / base_score
            fidelity = ratio / (ratio + 1.0)

            return float(np.clip(fidelity, 0.0, 1.0))

        except Exception:
            return 0.5

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
        raw = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
        return max(0.0, raw)