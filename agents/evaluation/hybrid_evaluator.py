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

    # ── Penalización drástica por exp_quality baja sostenida ────────────────────
    # Si un agente lleva N iters consecutivas con exp_quality < umbral,
    # su score se multiplica por un factor severo → force_adjust orgánico.
    EXP_QUALITY_PENALTY_THRESHOLD = 0.55
    EXP_QUALITY_PENALTY_WINDOW    = 5
    EXP_QUALITY_PENALTY_FACTOR    = 0.45

    def __init__(
        self,
        w_pred=0.10,       # última prioridad: no castigar disenso informado
        w_conf=0.05,       # auxiliar
        w_exp=0.40,        # primera prioridad: calidad explicaciones
        w_perf=0.20,       # cuarta: accuracy/f1 global
        w_instance=0.25,   # tercera: coherencia local + agreement implícito
        background_data=None
    ):
        self.w_pred     = w_pred
        self.w_conf     = w_conf
        self.w_exp      = w_exp
        self.w_perf     = w_perf
        self.w_instance = w_instance

        # Normalizar pesos
        total = w_pred + w_conf + w_exp + w_perf + w_instance
        self.w_pred     /= total
        self.w_conf     /= total
        self.w_exp      /= total
        self.w_perf     /= total
        self.w_instance /= total

        # Historial de exp_quality baja por agente (índice posicional)
        # {agent_idx → racha_consecutiva_baja}
        self._low_quality_streak = {}

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

        # ── Weighted majority: pondera por accuracy x confidence ─────────
        # Un agente con alta accuracy y alta confianza tiene mas peso
        # que varios agentes mediocres que coinciden. Crucial en instancias
        # frontera donde la mayoria simple es poco fiable.
        accs     = [m.get("accuracy", 0.5) for m in perfs]
        # exp_quality no disponible aún en este punto — se calcula después.
        # Usamos fidelity del historial previo si está disponible, o 0.5.
        prev_fids = [
            r.get("metrics", {}).get("exp_fidelity", 0.5)
            for r in responses
        ]
        majority = self._weighted_majority_vote(preds, accs, confs, prev_fids)

        # ── S_pred: alineación con mayoría ponderada por confianza ────────
        # Si acierto con alta confianza → premio extra
        # Si fallo con alta confianza   → penalización extra
        # Si fallo con baja confianza   → penalización suave (el modelo duda)
        S_pred = self._compute_S_pred(preds, confs, majority)

        S_conf = np.array(self._normalize(confs))
        perfs_prev = [r.get("metrics_prev", None) for r in responses]
        S_perf = np.array(self._performance_scores(perfs, perfs_prev))

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

        # ── Penalización drástica por exp_quality baja sostenida ─────────────
        # Si un agente lleva EXP_QUALITY_PENALTY_WINDOW iteraciones consecutivas
        # con exp_quality < EXP_QUALITY_PENALTY_THRESHOLD, se aplica
        # EXP_QUALITY_PENALTY_FACTOR sobre su score final.
        # Esto permite que force_adjust emerja orgánicamente sin reglas ad hoc,
        # respetando la jerarquía: exp_quality > acc_global > pred_local.
        agent_ids = [r.get("agent_id", i) for i, r in enumerate(responses)]
        for i, (aid, eq) in enumerate(zip(agent_ids, S_exp)):
            streak = self._low_quality_streak.get(aid, 0)
            if eq < self.EXP_QUALITY_PENALTY_THRESHOLD:
                streak += 1
            else:
                streak = 0
            self._low_quality_streak[aid] = streak

            if streak >= self.EXP_QUALITY_PENALTY_WINDOW:
                import logging
                logging.debug(
                    f"[HybridEvaluator] {aid}: exp_quality baja x{streak} iters "
                    f"(eq={eq:.3f}) → penalización drástica ×{self.EXP_QUALITY_PENALTY_FACTOR}"
                )
                scores[i] *= self.EXP_QUALITY_PENALTY_FACTOR

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

    def _get_delta(self, feature_idx, n_features, instance=None, sign=None):
        """
        Delta adaptativo al contexto de la instancia.

        Para features con std bajo (< 0.3), std*1.5 puede ser insuficiente
        para cruzar umbrales de activación en redes neuronales. Se usa un
        factor mínimo de 0.5 unidades absolutas.

        Si la instancia está cerca del límite del rango en la dirección de
        perturbación, se reduce el delta para no salir de la distribución real.
        """
        if self._feature_std is not None and feature_idx < len(self._feature_std):
            base_delta = float(self._feature_std[feature_idx] * 2.0)   # factor 2x (antes 1.5x)
            # mínimo absoluto para features con std bajo
            base_delta = max(base_delta, 0.3)
        else:
            base_delta = 0.5

        # Ajuste contextual: si la instancia está en el extremo del rango,
        # limitar el delta en esa dirección para no salir de la distribución.
        if (instance is not None
                and sign is not None
                and self._background is not None
                and feature_idx < instance.shape[-1]):
            feat_val = float(instance.flatten()[feature_idx])
            feat_min = float(self._background[:, feature_idx].min())
            feat_max = float(self._background[:, feature_idx].max())
            feat_range = feat_max - feat_min + 1e-8
            # Distancia al límite en la dirección de perturbación
            if sign > 0:
                margin = feat_max - feat_val
            else:
                margin = feat_val - feat_min
            # Si el margen es menor que el delta, usar el margen (con 10% de tolerancia)
            if margin < base_delta:
                base_delta = max(margin * 0.9, feat_range * 0.05)

        return float(base_delta)

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
                self._compute_fidelity(v, inst, model, pred, explainer_name=name)
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

            # ── Penalización suave por desacuerdo informado ───────────────
            # Si los explainers tienen fidelity alta de forma independiente
            # pero no coinciden en features, el desacuerdo refleja que el
            # modelo genuinamente tiene múltiples features relevantes desde
            # perspectivas distintas (gradientes vs sensibilidad local).
            # En ese caso penalizamos menos que si el desacuerdo viniera
            # de explicaciones de baja calidad.
            per_exp_fids = [
                per_explainer[name]["fidelity"][i]
                for name in explainer_names
                if name in per_explainer
                   and explanations_by_type[name][i] is not None
            ]
            mean_per_exp_fid = float(np.mean(per_exp_fids)) if per_exp_fids else f
            # Desacuerdo informado: fidelity individual alta pero agreement bajo
            informed_disagreement = mean_per_exp_fid > 0.65 and a < 0.35
            # Factor de moderación: si el desacuerdo es informado, a_effective
            # se eleva parcialmente hacia 0.5 en lugar de castigar con a real.
            a_effective = (0.5 * a + 0.5 * 0.45) if informed_disagreement else a

            if has_stability:
                s = float(mean_stability_arr[i])
                q = 0.30 * c + 0.25 * s + 0.30 * f + 0.15 * a_effective
            else:
                q = 0.40 * c + 0.40 * f + 0.20 * a_effective
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
        """
        Acuerdo inter-explainer con similitud ponderada por rango.

        Reemplaza Jaccard top-k binario por una métrica gradual que da
        crédito parcial cuando los explainers identifican features distintas
        pero igualmente válidas — caso típico cuando SHAP mide gradientes
        internos y LIME mide sensibilidad local de la frontera.

        Método: rank-weighted overlap.
          - Cada feature recibe peso 1/rank (rank 1 = top feature).
          - Score = Σ min(w_a[f], w_b[f]) / Σ max(w_a[f], w_b[f])
          - Si ambos coinciden en top-1 → score alto.
          - Si identifican features distintas pero ambas con fidelity alta
            → score medio (no cero), reflejando incertidumbre legítima.
          - Score=0 solo si no comparten ninguna feature en top-k.
        """
        explainer_names = list(explanations_by_type.keys())

        if len(explainer_names) < 2:
            return [1.0] * n

        def rank_weights(v, k=5):
            """Devuelve dict {feature_idx: 1/rank} para top-k features."""
            indices = np.argsort(np.abs(v))[::-1][:k]
            weights = [1.0, 0.5, 0.25, 0.125, 0.0625]  # decaimiento geométrico
            return {int(idx): weights[rank] for rank, idx in enumerate(indices)}

        agreement = []
        for i in range(n):
            weighted_vecs = []
            for name in explainer_names:
                v = explanations_by_type[name][i]
                if v is not None and np.linalg.norm(v) > 1e-8:
                    weighted_vecs.append(rank_weights(v, k=5))

            if len(weighted_vecs) < 2:
                agreement.append(0.5)
                continue

            pairs = scores_sum = 0
            for a in range(len(weighted_vecs)):
                for b in range(a + 1, len(weighted_vecs)):
                    pairs += 1
                    all_feats = set(weighted_vecs[a]) | set(weighted_vecs[b])
                    numerator   = sum(min(weighted_vecs[a].get(f, 0),
                                         weighted_vecs[b].get(f, 0))
                                      for f in all_feats)
                    denominator = sum(max(weighted_vecs[a].get(f, 0),
                                         weighted_vecs[b].get(f, 0))
                                      for f in all_feats)
                    scores_sum += numerator / denominator if denominator > 0 else 0.0

            agreement.append(scores_sum / pairs if pairs > 0 else 0.5)

        return agreement

    # ======================================================
    # Fidelity — perturbación escala-aware con background
    # ======================================================

    def _compute_fidelity(self, expl_vector, instance, model, original_pred,
                          explainer_name=""):
        """
        Fidelity dual según tipo de explainer:

        SHAP  → correlación de gradientes: mide si el vector SHAP predice
                correctamente la dirección del cambio de probabilidad al
                perturbar cada feature. No depende de cruzar la frontera
                de decisión — mide alineación entre gradientes del explainer
                y gradientes numéricos del modelo. Apropiado para zonas de
                frontera donde cambiar la clase predicha requiere perturbaciones
                grandes y los vecinos son de clases mezcladas.

        LIME  → sensibilidad local con probe set de vecinos (método actual):
                mide si las top-features de LIME realmente cambian la predicción.
                Funciona bien para LIME porque éste ya está calibrado localmente.
        """
        if expl_vector is None or instance is None or model is None:
            return 0.5

        try:
            if "shap" in explainer_name.lower():
                return self._shap_fidelity_gradient(expl_vector, instance, model)
            else:
                return self._lime_fidelity_perturbation(expl_vector, instance, model, original_pred)
        except Exception:
            return 0.5

    def _shap_fidelity_gradient(self, shap_vector, instance, model):
        """
        Fidelity SHAP via correlación con gradientes numéricos.

        Método:
        1. Calcular gradiente numérico de P(clase_predicha) respecto a
           cada feature, en la instancia concreta y en un subconjunto
           del background cercano.
        2. Medir correlación de Spearman entre |shap_vector| y |gradiente|.
           Spearman captura coincidencia en el ranking de features sin
           asumir linealidad.
        3. Normalizar a [0, 1]: fidelity = (corr + 1) / 2.

        Un SHAP de alta fidelidad predice el mismo ranking de features
        que los gradientes numéricos reales del modelo.
        """
        from scipy.stats import spearmanr

        v    = np.asarray(shap_vector, dtype=float)
        inst = np.asarray(instance, dtype=float).reshape(1, -1)
        n_feat = v.shape[0]

        pred_class = int(model.predict(inst)[0])

        # ── Puntos de evaluación: instancia + vecinos más próximos ───────────
        if self._background is not None and len(self._background) >= 3:
            std_all   = self._feature_std if self._feature_std is not None else np.ones(n_feat)
            norm_bg   = (self._background - inst) / (std_all + 1e-8)
            dists     = np.linalg.norm(norm_bg, axis=1)
            # Solo usar vecinos de la misma clase predicha para evitar
            # gradientes en dirección opuesta que confunden la correlación
            same_class_mask = np.array([
                int(model.predict(self._background[i:i+1])[0]) == pred_class
                for i in range(len(self._background))
            ])
            if same_class_mask.sum() >= 3:
                dists_filtered = np.where(same_class_mask, dists, np.inf)
            else:
                dists_filtered = dists  # fallback: todos
            n_neighbors = min(5, int(same_class_mask.sum()))
            neighbor_idxs = np.argsort(dists_filtered)[:n_neighbors]
            probe_points = np.vstack([inst, self._background[neighbor_idxs]])
        else:
            probe_points = inst

        # ── Gradiente numérico: ΔP(pred_class) / Δfeature_i ─────────────────
        grad_sum = np.zeros(n_feat)
        n_valid  = 0

        for x_probe in probe_points:
            x = x_probe.reshape(1, -1)
            try:
                p0 = model.predict_proba(x)[0][pred_class]
            except Exception:
                continue

            grad_local = np.zeros(n_feat)
            for fi in range(n_feat):
                delta = self._get_delta(fi, n_feat, instance=x, sign=+1)
                xi_p = x.copy(); xi_p[0, fi] += delta
                xi_m = x.copy(); xi_m[0, fi] -= delta
                try:
                    pp = model.predict_proba(xi_p)[0][pred_class]
                    pm = model.predict_proba(xi_m)[0][pred_class]
                    grad_local[fi] = (pp - pm) / (2 * delta + 1e-12)
                except Exception:
                    grad_local[fi] = 0.0

            grad_sum += np.abs(grad_local)
            n_valid  += 1

        if n_valid == 0:
            return 0.5

        grad_mean = grad_sum / n_valid

        # ── Correlación de Spearman entre SHAP y gradiente (con signo) ───────
        # Usar vectores con signo: SHAP correcto debe tener el mismo signo
        # que el gradiente de P(clase_predicha) respecto a cada feature.
        # spearmanr(shap, grad) ≈ 1.0 si coinciden en ranking Y dirección,
        # ≈ 0.0 si son ortogonales, ≈ -1.0 si son opuestos.
        if np.std(v) < 1e-8 or np.std(grad_mean) < 1e-8:
            return 0.5   # vectores constantes — no hay ranking que comparar

        corr, _ = spearmanr(v, grad_mean)
        if np.isnan(corr):
            return 0.5

        # Normalizar [-1, 1] → [0, 1]
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def _lime_fidelity_perturbation(self, expl_vector, instance, model, original_pred):
        """
        Fidelity LIME: sensibilidad local con probe set de vecinos.
        Versión actual — funciona bien para LIME porque éste ya está
        calibrado en el vecindario local de la instancia.
        """
        v          = np.asarray(expl_vector, dtype=float)
        n_features = v.shape[0]
        inst       = np.asarray(instance, dtype=float).reshape(1, -1)

        k     = max(1, int(0.3 * n_features))
        top_k = list(np.argsort(np.abs(v))[::-1][:k].tolist())
        rest  = [i for i in range(n_features) if i not in top_k]

        if not rest:
            return 0.5

        if self._background is not None and len(self._background) >= 5:
            norm_bg       = (self._background - inst) / (self._feature_std + 1e-8)
            dists         = np.linalg.norm(norm_bg, axis=1)
            neighbor_idxs = np.argsort(dists)[:9]
            probe_set     = np.vstack([inst, self._background[neighbor_idxs]])
        else:
            probe_set = inst

        def mean_sensitivity(feature_indices):
            effects = []
            for x_probe in probe_set:
                x  = x_probe.reshape(1, -1)
                y0 = int(model.predict(x)[0])
                changed = False
                for fi in feature_indices:
                    grad_sign = np.sign(v[fi]) if abs(v[fi]) > 1e-8 else 0
                    signs = ([-grad_sign, grad_sign] if grad_sign != 0 else [+1, -1])
                    for sign in signs:
                        delta = self._get_delta(fi, n_features, instance=x, sign=sign)
                        xi = x.copy()
                        xi[0, fi] += sign * delta
                        if int(model.predict(xi)[0]) != y0:
                            effects.append(1.0)
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    effects.append(0.0)
            return float(np.mean(effects)) if effects else 0.0

        top_score  = mean_sensitivity(top_k)
        base_idxs  = np.random.choice(rest, size=min(k, len(rest)), replace=False)
        base_score = mean_sensitivity(base_idxs.tolist()) + 1e-6
        ratio      = top_score / base_score
        return float(np.clip(ratio / (ratio + 1.0), 0.0, 1.0))

    # ======================================================
    # Métricas clásicas
    # ======================================================

    def _performance_scores(self, metrics, metrics_prev=None):
        """
        Score de rendimiento con penalización por degradación.

        Base:     0.4·f1_macro + 0.3·accuracy + 0.2·precision_w + 0.1·recall_w
        Penalty:  por cada métrica que baja respecto a la iteración anterior,
                  se aplica un descuento proporcional a la caída.
                  La penalización total se escala por severidad:
                    - caída < 0.02  → penalización suave  (×1.0)
                    - caída < 0.05  → penalización media   (×2.0)
                    - caída ≥ 0.05  → penalización severa  (×3.5)
        """
        _METRIC_WEIGHTS = {
            "accuracy":  0.30,
            "f1":        0.40,
            "precision": 0.20,
            "recall":    0.10,
        }
        _SEVERITY = [(0.02, 1.0), (0.05, 2.0), (float("inf"), 3.5)]

        if metrics_prev is None:
            metrics_prev = [None] * len(metrics)

        scores = []
        for m, m_prev in zip(metrics, metrics_prev):
            if not m:
                scores.append(0.0)
                continue

            base = sum(w * m.get(k, 0.0) for k, w in _METRIC_WEIGHTS.items())

            penalty = 0.0
            if m_prev:
                for k, w in _METRIC_WEIGHTS.items():
                    delta = m_prev.get(k, 0.0) - m.get(k, 0.0)  # positivo = degradación
                    if delta > 0:
                        # escalar según severidad
                        scale = next(s for thr, s in _SEVERITY if delta < thr)
                        penalty += w * delta * scale

            score = max(0.0, base - penalty)
            if m_prev and penalty > 0:
                import logging
                logging.debug(
                    f"[S_perf] degradación detectada | "
                    f"base={base:.4f} penalty={penalty:.4f} score={score:.4f}"
                )
            scores.append(score)
        return scores

    # ======================================================
    # Utils
    # ======================================================

    def _majority_vote(self, preds):
        return max(set(preds), key=preds.count)

    def _weighted_majority_vote(self, preds, accs, confs, fids=None):
        """
        Voto ponderado por accuracy × confidence × fidelity_explicativa.

        Incorporar fidelity evita que un modelo con acc alta pero explicaciones
        poco fieles domine el voto. Un modelo que predice bien pero no puede
        justificar su predicción (fidelity baja) recibe menos peso.

        Peso = acc × conf × fidelity_factor
        donde fidelity_factor = 0.5 + 0.5 × fidelity  (rango [0.5, 1.0])
        — factor suave para no penalizar demasiado en la primera iteración.

        Si todos los pesos son iguales (eps), cae al conteo simple.
        """
        if fids is None:
            fids = [0.5] * len(preds)

        vote_weights = {}
        for pred, acc, conf, fid in zip(preds, accs, confs, fids):
            # fidelity_factor en [0.5, 1.0] — penalización suave
            fid_factor = 0.5 + 0.5 * float(fid)
            w = float(acc) * float(conf) * fid_factor
            vote_weights[pred] = vote_weights.get(pred, 0.0) + w

        total = sum(vote_weights.values())
        if total < 1e-8:
            return max(set(preds), key=preds.count)

        winner = max(vote_weights, key=vote_weights.get)

        # Log informativo cuando el ganador difiere de la mayoria simple
        simple = max(set(preds), key=preds.count)
        if winner != simple:
            pesos = {pred: round(weight / total, 2) for pred, weight in vote_weights.items()}
            print(f"[WeightedVote] Mayoria simple={simple} | "
                f"Mayoria ponderada={winner} | "
                f"pesos={pesos}")

        return winner

    def _normalize(self, values, eps=1e-8):
        v = np.array(values, dtype=float)
        if v.max() - v.min() < eps:
            return v.tolist()
        return ((v - v.min()) / (v.max() - v.min() + eps)).tolist()

    def _cosine(self, a, b, eps=1e-8):
        raw = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
        return max(0.0, raw)


    