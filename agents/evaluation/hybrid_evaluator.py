import numpy as np


class HybridEvaluator:
    """
    Evaluador híbrido multi-métrica y explanation-aware.

    Métricas explicativas calculadas POR SEPARADO para cada explainer:
    - consensus:  mínimo de similitud coseno inter-agente entre explainers
    - stability:  estabilidad temporal usando historial por tipo de explainer
    - fidelity:   sensibilidad de las top-features vs features aleatorias (escala-aware)
    - agreement:  acuerdo entre explainers del mismo agente (SHAP vs LIME)
    - quality:    combinación ponderada de las cuatro anteriores

    Métricas de predicción sobre la instancia concreta:
    - S_pred:     alineación con mayoría, ponderada por confianza relativa.
                  Si discrepas con alta confianza, la penalización es mayor.
    - S_instance: coherencia entre la explicación del agente y su predicción
                  local. Mide si las top-features del agente realmente cambian
                  la predicción al ser perturbadas, usando solo la instancia
                  dada (no el background), y penaliza si la predicción local
                  difiere de la mayoría.
    """

    def __init__(
        self,
        w_pred=0.25,
        w_conf=0.10,
        w_exp=0.30,
        w_perf=0.20,
        w_instance=0.15,       # ← nuevo peso para S_instance
        background_data=None
    ):
        self.w_pred     = w_pred
        self.w_conf     = w_conf
        self.w_exp      = w_exp
        self.w_perf     = w_perf
        self.w_instance = w_instance

        # Normalizar pesos por si acaso no suman 1
        total = w_pred + w_conf + w_exp + w_perf + w_instance
        self.w_pred     /= total
        self.w_conf     /= total
        self.w_exp      /= total
        self.w_perf     /= total
        self.w_instance /= total

        if background_data is not None:
            std = np.std(background_data, axis=0)
            self._feature_std = np.where(std > 1e-6, std, 1.0)
            self._background  = np.asarray(background_data, dtype=float)
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

        # ── S_pred: alineación con mayoría ponderada por confianza ────────
        # Si acierto con alta confianza → premio extra
        # Si fallo con alta confianza   → penalización extra
        # Si fallo con baja confianza   → penalización suave (el modelo duda)
        S_pred = self._compute_S_pred(preds, confs, majority)

        S_conf = np.array(self._normalize(confs))
        S_perf = np.array(self._performance_scores(perfs))

        exp_result = self._explanation_quality_multi(
            explanations_by_type, histories_by_type, instances, models, preds, responses
        )
        S_exp = exp_result["quality"]

        # ── S_instance: coherencia explicación-predicción local ───────────
        S_instance = self._compute_S_instance(
            responses, explanations_by_type, instances, models, preds, majority
        )

        scores = (
            self.w_pred     * S_pred     +
            self.w_conf     * S_conf     +
            self.w_exp      * S_exp      +
            self.w_perf     * S_perf     +
            self.w_instance * S_instance
        )

        S_acc = np.array([m.get("accuracy", 0.0) for m in perfs])

        return {
            "global_score":        float(scores.mean()),
            "majority_prediction": majority,
            "scores":              scores.tolist(),
            "components": {
                "pred":     S_pred.tolist(),
                "conf":     S_conf.tolist(),
                "perf":     S_perf.tolist(),
                "exp":      S_exp.tolist(),
                "instance": S_instance.tolist(),
                "acc":      S_acc.tolist(),
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
    # S_pred — alineación con mayoría ponderada por confianza
    # ======================================================

    def _compute_S_pred(self, preds, confs, majority):
        """
        Para cada agente:
        - Acierta (pred == majority):
            score = 0.5 + 0.5 * conf_normalizada
            → entre 0.5 (baja confianza) y 1.0 (alta confianza)
        - Falla (pred != majority):
            score = 0.5 - 0.5 * conf_normalizada
            → entre 0.5 (baja confianza, duda) y 0.0 (alta confianza, error convencido)

        La confianza se normaliza respecto al grupo para que sea relativa.
        """
        confs_arr = np.array(confs, dtype=float)

        # Normalizar confianza en [0, 1] dentro del grupo
        c_min, c_max = confs_arr.min(), confs_arr.max()
        if c_max - c_min < 1e-8:
            confs_norm = np.full_like(confs_arr, 0.5)
        else:
            confs_norm = (confs_arr - c_min) / (c_max - c_min)

        scores = []
        for pred, c in zip(preds, confs_norm):
            if pred == majority:
                scores.append(0.5 + 0.5 * c)
            else:
                scores.append(0.5 - 0.5 * c)

        return np.array(scores)

    # ======================================================
    # S_instance — coherencia explicación-predicción local
    # ======================================================

    def _compute_S_instance(self, responses, explanations_by_type,
                             instances, models, preds, majority):
        """
        Para cada agente combina dos señales:

        1. local_fidelity: ¿las top-features de la explicación realmente
           cambian la predicción al perturbar SOLO la instancia dada?
           (más rápido y directo que el fidelity del background)

        2. pred_penalty: si el agente discrepa de la mayoría, se penaliza
           proporcionalmente a cuánto discrepa en probabilidad.
           Si coincide con la mayoría, no hay penalización.

        score_final = local_fidelity * pred_penalty
        """
        n = len(responses)
        S_instance = np.zeros(n)

        explainer_names = list(explanations_by_type.keys())

        for i, (response, instance, model, pred) in enumerate(
            zip(responses, instances, models, preds)
        ):
            # ── 1. Local fidelity sobre la instancia concreta ─────────────
            # Promedio de fidelity local entre todos los explainers del agente
            local_fids = []
            for name in explainer_names:
                v = explanations_by_type[name][i]
                if v is not None:
                    lf = self._local_fidelity(v, instance, model, pred)
                    local_fids.append(lf)

            local_fidelity = float(np.mean(local_fids)) if local_fids else 0.5

            # ── 2. Pred penalty basada en probabilidad ────────────────────
            # Si el modelo tiene predict_proba, usamos la probabilidad de la
            # clase mayoritaria como señal de "qué tan equivocado está".
            # Si no tiene proba, usamos señal binaria suavizada.
            pred_penalty = self._pred_alignment_score(
                model, instance, pred, majority
            )

            S_instance[i] = local_fidelity * pred_penalty

        return S_instance

    def _local_fidelity(self, expl_vector, instance, model, pred):
        """
        Fidelity local: perturba SOLO la instancia dada (no el background).
        Más sensible a lo que le pasa a este punto concreto.
        """
        if expl_vector is None or instance is None or model is None:
            return 0.5
        try:
            v          = np.asarray(expl_vector, dtype=float)
            n_features = v.shape[0]
            x          = np.asarray(instance, dtype=float).reshape(1, -1)

            k     = max(1, int(0.3 * n_features))
            top_k = np.argsort(np.abs(v))[::-1][:k].tolist()
            rest  = [j for j in range(n_features) if j not in top_k]

            if not rest:
                return 0.5

            def sensitivity(feature_indices):
                flips = 0
                for fi in feature_indices:
                    delta = self._get_delta(fi, n_features)
                    for sign in (+1, -1):
                        xi = x.copy()
                        xi[0, fi] += sign * delta
                        if int(model.predict(xi)[0]) != pred:
                            flips += 1
                            break
                return flips / len(feature_indices)

            top_score  = sensitivity(top_k)
            base_idxs  = np.random.choice(rest, size=min(k, len(rest)), replace=False)
            base_score = sensitivity(base_idxs.tolist()) + 1e-6

            ratio = top_score / base_score
            return float(np.clip(ratio / (ratio + 1.0), 0.0, 1.0))

        except Exception:
            return 0.5

    def _pred_alignment_score(self, model, instance, pred, majority):
        """
        Devuelve un score [0, 1] que refleja cuánto se alinea la predicción
        del agente con la mayoría, usando probabilidades si están disponibles.

        - pred == majority con alta proba de majority → cerca de 1.0
        - pred != majority con alta proba de pred (equivocado convencido) → cerca de 0.0
        - pred != majority con baja proba (duda razonable) → ~0.4
        """
        if model is None or instance is None:
            return 1.0 if pred == majority else 0.0

        try:
            x = np.asarray(instance, dtype=float).reshape(1, -1)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)[0]

                # Índice de la clase mayoritaria dentro del vector de proba
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    class_list = list(classes)
                    if majority in class_list:
                        maj_idx  = class_list.index(majority)
                        pred_idx = class_list.index(pred) if pred in class_list else None
                    else:
                        # fallback si majority no está en classes_
                        return 1.0 if pred == majority else 0.3
                else:
                    # Sin classes_, usar majority/pred como índice directo
                    maj_idx  = majority
                    pred_idx = pred

                p_majority = float(proba[maj_idx])

                if pred == majority:
                    # Acierta: score proporcional a la confianza en la clase correcta
                    return float(np.clip(0.5 + 0.5 * p_majority, 0.5, 1.0))
                else:
                    # Falla: penalización proporcional a la confianza en la clase errónea
                    p_pred = float(proba[pred_idx]) if pred_idx is not None else 1.0 - p_majority
                    # Cuanto más convencido del error, peor score
                    return float(np.clip(p_majority - p_pred + 0.5, 0.0, 0.8))

            else:
                # Sin probabilidades: binario suavizado
                return 1.0 if pred == majority else 0.3

        except Exception:
            return 1.0 if pred == majority else 0.3

    # ======================================================
    # Feature std estimator
    # ======================================================

    def _get_delta(self, feature_idx, n_features):
        if self._feature_std is not None and feature_idx < len(self._feature_std):
            return float(self._feature_std[feature_idx] * 1.5)
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
        self, explanations_by_type, histories_by_type, instances, models, preds,
        responses
    ):
        n = len(histories_by_type)
        explainer_names = list(explanations_by_type.keys())

        iters_since_adjust = [r.get("iters_since_adjust", 99) for r in responses]

        per_explainer           = {}
        consensus_per_explainer = {}
        all_stability           = np.zeros(n)
        all_fidelity            = np.zeros(n)
        valid_count             = np.zeros(n)
        has_valid               = np.zeros(n, dtype=bool)

        for name in explainer_names:
            vectors = explanations_by_type[name]
            valid   = [v for v in vectors if v is not None]

            if not valid:
                per_explainer[name] = {
                    "consensus": [0.0] * n,
                    "stability": [None] * n,
                    "fidelity":  [0.5] * n,
                }
                consensus_per_explainer[name] = [0.0] * n
                continue

            center    = np.mean(valid, axis=0)
            consensus = [
                self._cosine(v, center) if v is not None else 0.0
                for v in vectors
            ]
            consensus_per_explainer[name] = consensus

            stability = []
            for i, (v, hist_by_type) in enumerate(zip(vectors, histories_by_type)):
                hist = hist_by_type.get(name, [])
                if v is None or len(hist) == 0:
                    stability.append(None)
                else:
                    raw_stab = self._cosine(v, np.mean(hist, axis=0))
                    if iters_since_adjust[i] <= 2:
                        stability.append(max(raw_stab, 0.75))
                    else:
                        stability.append(raw_stab)

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
                    all_stability[i] += stability[i] if stability[i] is not None else 0.0
                    all_fidelity[i]  += fidelity[i]
                    valid_count[i]   += 1
                    has_valid[i]      = True

        min_consensus = np.ones(n)
        for name, cons_list in consensus_per_explainer.items():
            for i, c in enumerate(cons_list):
                if explanations_by_type[name][i] is not None:
                    min_consensus[i] = min(min_consensus[i], c)
        mean_consensus = min_consensus.tolist()

        safe_count         = np.where(valid_count > 0, valid_count, 1)
        mean_stability_arr = all_stability / safe_count
        mean_fidelity      = (all_fidelity / safe_count).tolist()

        agreement = self._compute_agreement(explanations_by_type, n)

        quality = []
        for i, (c, f, a) in enumerate(zip(mean_consensus, mean_fidelity, agreement)):
            has_stability = any(
                per_explainer[name]["stability"][i] is not None
                for name in explainer_names
                if name in per_explainer
                    and explanations_by_type[name][i] is not None
            )
            if has_stability:
                s = float(mean_stability_arr[i])
                q = 0.30 * c + 0.25 * s + 0.30 * f + 0.15 * a
            else:
                q = 0.40 * c + 0.40 * f + 0.20 * a
            quality.append(q)

        quality = np.array(quality)

        mean_stability_list = [
            float(mean_stability_arr[i]) if has_valid[i] else 0.5
            for i in range(n)
        ]

        return {
            "consensus":     mean_consensus,
            "stability":     mean_stability_list,
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
            top_features_per_explainer = []
            for name in explainer_names:
                v = explanations_by_type[name][i]
                if v is not None and np.linalg.norm(v) > 1e-8:
                    k     = min(3, len(v))
                    top_k = set(np.argsort(np.abs(v))[::-1][:k].tolist())
                    top_features_per_explainer.append(top_k)

            if len(top_features_per_explainer) < 2:
                agreement.append(0.5)
                continue

            pairs = scores_sum = 0
            for a in range(len(top_features_per_explainer)):
                for b in range(a + 1, len(top_features_per_explainer)):
                    pairs += 1
                    intersection = len(
                        top_features_per_explainer[a] & top_features_per_explainer[b])
                    union = len(
                        top_features_per_explainer[a] | top_features_per_explainer[b])
                    scores_sum += intersection / union if union > 0 else 0.0

            agreement.append(scores_sum / pairs if pairs > 0 else 0.5)

        return agreement

    # ======================================================
    # Fidelity — perturbación escala-aware con background
    # ======================================================

    def _compute_fidelity(self, expl_vector, instance, model, original_pred):
        if expl_vector is None or instance is None or model is None:
            return 0.5

        try:
            v          = np.asarray(expl_vector, dtype=float)
            n_features = v.shape[0]

            k     = max(1, int(0.3 * n_features))
            top_k = set(np.argsort(np.abs(v))[::-1][:k].tolist())
            rest  = [i for i in range(n_features) if i not in top_k]

            if not rest:
                return 0.5

            if self._background is not None and len(self._background) >= 5:
                idxs      = np.random.choice(len(self._background), size=20, replace=False)
                probe_set = self._background[idxs]
            else:
                probe_set = np.asarray(instance, dtype=float).reshape(1, -1)

            def mean_sensitivity(feature_indices):
                effects = []
                for x_probe in probe_set:
                    x  = x_probe.reshape(1, -1)
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
            base_score = mean_sensitivity(base_idxs.tolist()) + 1e-6

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
        return scores

    # ======================================================
    # Utils
    # ======================================================

    def _majority_vote(self, preds):
        return max(set(preds), key=preds.count)

    def _normalize(self, values, eps=1e-8):
        v = np.array(values, dtype=float)
        if v.max() - v.min() < eps:
            return v.tolist()
        return ((v - v.min()) / (v.max() - v.min() + eps)).tolist()

    def _cosine(self, a, b, eps=1e-8):
        raw = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
        return max(0.0, raw)