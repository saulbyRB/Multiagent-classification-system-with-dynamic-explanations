import numpy as np


# ── Umbrales para ser mentor ───────────────────────────────────────────────────
MENTOR_THRESHOLDS = {
    "exp_quality": 0.65,   # recalibrado: rank-weighted agreement da valores menores
    "agreement":   0.25,   # recalibrado: desacuerdo SHAP/LIME informado da ~0.27-0.35
    "fidelity":    0.55,
    "confidence":  0.60,
}

# ── Umbrales para mentor disidente (contradice la mayoría) ───────────────────
# Un agente que discrepa puede ser mentor si sus explicaciones son
# suficientemente buenas — su disenso es informado, no ruido.
# fidelity es el criterio más importante: si el modelo realmente usa las
# features que explica, su predicción disidente merece consideración.
MENTOR_DISSENT_THRESHOLDS = {
    "exp_quality": 0.72,   # ligeramente por debajo del umbral normal
    "fidelity":    0.65,   # sigue siendo exigente — disenso bien fundado
    "confidence":  0.65,   # confianza mínima en la predicción disidente
}

MENTOR_TIE_TOLERANCE = 1e-4

MENTOR_SCORE_WEIGHTS = {
    "exp_quality": 0.35,
    "agreement":   0.25,
    "fidelity":    0.25,
    "confidence":  0.15,
}


class FeedbackBuilder:

    def build(self, agent_id, decision, evaluation, explanations=None, idx=0):
        scores      = evaluation["scores"]
        n           = len(scores)
        my_score    = scores[idx]
        peer_scores = [s for i, s in enumerate(scores) if i != idx]

        exp_detail  = evaluation["components"].get("exp_detail", {})
        quality_arr = exp_detail.get("quality",   [0.5] * n)
        stab_arr    = exp_detail.get("stability", [0.5] * n)

        if len(peer_scores) > 0:
            mean_peer    = float(np.mean(peer_scores))
            relative_pos = float(np.tanh((my_score - mean_peer) * 5))
        else:
            mean_peer    = my_score
            relative_pos = 0.0

        all_peers_struggling = (
            len(peer_scores) > 0 and
            all(s < 0.5 for s in peer_scores)
        )

        group_pressure = float(my_score - mean_peer)

        return {
            "strategy":    decision,
            "agent_id":    agent_id,
            "evaluation": {
                "reward":                 my_score,
                "prediction_alignment":   evaluation["components"]["pred"][idx],
                "confidence":             evaluation["components"]["conf"][idx],
                "explanation_similarity": evaluation["components"]["exp"][idx],
                "exp_quality":            quality_arr[idx],
                "exp_consensus":          exp_detail.get("consensus", [0.5]*n)[idx],
                "exp_stability":          stab_arr[idx],
                "num_explainers":         len(explanations) if explanations else 0,
                "global_score":           evaluation["global_score"],
            },
            "group_signals": {
                "peer_scores":          peer_scores,
                "mean_peer_score":      mean_peer,
                "relative_position":    relative_pos,
                "group_pressure":       group_pressure,
                "all_peers_struggling": all_peers_struggling,
                "group_quality_mean":   float(np.mean(quality_arr)),
                "group_stability_mean": float(np.mean(stab_arr)),
            }
        }

    # ======================================================
    # Mentor
    # ======================================================

    def find_mentor(self, responses, evaluation):
        """
        Identifica el agente mentor entre todas las respuestas.

        Cambio respecto a la version anterior:
          - El filtro de agreement usa top-5 Jaccard (antes top-3).
            Esto es mas tolerante con diferencias metodologicas entre
            SHAP y LIME: ambos pueden identificar las mismas features
            relevantes pero en distinto orden o con distinto top-1.
            El umbral se baja a 0.40 (antes 0.60) acorde al cambio de escala.
          - El agreement general del evaluador (exp_quality, logs) sigue
            siendo top-3 Jaccard sin cambios.
        """
        majority    = evaluation["majority_prediction"]
        exp_detail  = evaluation["components"].get("exp_detail", {})
        n           = len(responses)

        quality_arr   = exp_detail.get("quality",   [0.5] * n)
        fidelity_arr  = exp_detail.get("fidelity",  [0.5] * n)
        conf_arr      = evaluation["components"].get("conf", [0.0] * n)
        preds         = [r["prediction"] for r in responses]

        # Agreement top-5 especifico para seleccion de mentor
        per_explainer   = exp_detail.get("per_explainer", {})
        mentor_agreement = self._compute_mentor_agreement(responses, per_explainer, n)

        # ── Filtrar candidatos alineados con la mayoría ─────────────────────
        candidates = []
        for i, r in enumerate(responses):
            if preds[i] != majority:
                continue
            if quality_arr[i]        < MENTOR_THRESHOLDS["exp_quality"]:
                continue
            if mentor_agreement[i]   < MENTOR_THRESHOLDS["agreement"]:
                continue
            if fidelity_arr[i]       < MENTOR_THRESHOLDS["fidelity"]:
                continue
            if conf_arr[i]           < MENTOR_THRESHOLDS["confidence"]:
                continue

            mentor_score = (
                MENTOR_SCORE_WEIGHTS["exp_quality"] * quality_arr[i]      +
                MENTOR_SCORE_WEIGHTS["agreement"]   * mentor_agreement[i]  +
                MENTOR_SCORE_WEIGHTS["fidelity"]    * fidelity_arr[i]      +
                MENTOR_SCORE_WEIGHTS["confidence"]  * conf_arr[i]
            )
            candidates.append((i, r["agent_id"], mentor_score, False))

        # ── Si no hay candidatos alineados, buscar mentor disidente ──────────
        # Un agente que contradice la mayoría puede ser mentor si sus
        # explicaciones son suficientemente fiables — su disenso es informado.
        is_dissent_mentor = False
        if not candidates:
            for i, r in enumerate(responses):
                if preds[i] == majority:
                    continue
                if quality_arr[i]  < MENTOR_DISSENT_THRESHOLDS["exp_quality"]:
                    continue
                if fidelity_arr[i] < MENTOR_DISSENT_THRESHOLDS["fidelity"]:
                    continue
                if conf_arr[i]     < MENTOR_DISSENT_THRESHOLDS["confidence"]:
                    continue

                mentor_score = (
                    MENTOR_SCORE_WEIGHTS["exp_quality"] * quality_arr[i]      +
                    MENTOR_SCORE_WEIGHTS["agreement"]   * mentor_agreement[i]  +
                    MENTOR_SCORE_WEIGHTS["fidelity"]    * fidelity_arr[i]      +
                    MENTOR_SCORE_WEIGHTS["confidence"]  * conf_arr[i]
                )
                candidates.append((i, r["agent_id"], mentor_score, True))

            if candidates:
                is_dissent_mentor = True
                print(f"[FeedbackBuilder] ⚠ Mentor DISIDENTE activado "
                      f"(ningún candidato alineado con mayoría superó umbrales)")

        if not candidates:
            return None, [], False

        # ── Seleccionar el mejor o empatados ───────────────────────────────
        best_score    = max(c[2] for c in candidates)
        mentors       = [
            (i, aid, s, d) for i, aid, s, d in candidates
            if abs(s - best_score) < MENTOR_TIE_TOLERANCE
        ]
        mentor_ids    = [aid for _, aid, _, _ in mentors]
        mentor_scores = np.array([s for _, _, s, _ in mentors])

        # ── Construir vector mentor ponderado ─────────────────────────────
        explainer_names = list(per_explainer.keys())

        mentor_vectors = []
        for i, aid, ms, is_diss in mentors:
            r            = responses[i]
            weighted_sum = None
            total_weight = 0.0

            for name in explainer_names:
                exp_data = per_explainer.get(name, {})
                fid_list = exp_data.get("fidelity", [0.5] * n)
                fid      = float(fid_list[i]) if i < len(fid_list) else 0.5

                v = self._get_explanation_vector(r, name)
                if v is None:
                    continue

                weighted_sum  = fid * v if weighted_sum is None else weighted_sum + fid * v
                total_weight += fid

            if weighted_sum is not None and total_weight > 1e-8:
                mentor_vectors.append((weighted_sum / total_weight, ms))

        if not mentor_vectors:
            return None, mentor_ids

        total_ms  = sum(ms for _, ms in mentor_vectors)
        final_vec = sum((ms / total_ms) * v for v, ms in mentor_vectors)

        norm = np.linalg.norm(final_vec)
        if norm > 1e-8:
            final_vec = final_vec / norm

        dissent_tag = " [DISIDENTE]" if is_dissent_mentor else ""
        print(f"[FeedbackBuilder] Mentor(s): {mentor_ids}{dissent_tag} | "
              f"score={best_score:.3f} | "
              f"agreement_top5={[round(mentor_agreement[i],2) for i,_,_,_ in mentors]} | "
              f"top3={self._top3_indices(final_vec)}")

        return final_vec, mentor_ids, is_dissent_mentor

    def build_with_mentor(self, agent_id, decision, evaluation,
                          explanations=None, idx=0,
                          mentor_vector=None, mentor_ids=None,
                          is_dissent_mentor=False):
        feedback  = self.build(
            agent_id=agent_id,
            decision=decision,
            evaluation=evaluation,
            explanations=explanations,
            idx=idx
        )
        is_mentor = agent_id in (mentor_ids or [])

        feedback["peer_guidance"] = {
            "has_mentor":         mentor_vector is not None and not is_mentor,
            "is_mentor":          is_mentor,
            "mentor_ids":         mentor_ids or [],
            "mentor_vector":      mentor_vector.tolist() if (
                mentor_vector is not None and not is_mentor
            ) else None,
            "is_dissent_mentor":  is_dissent_mentor and not is_mentor,
        }
        return feedback

    # ======================================================
    # Agreement top-5 para seleccion de mentor
    # ======================================================

    def _compute_mentor_agreement(self, responses, per_explainer, n):
        """
        Rank-weighted overlap entre explainers del mismo agente.

        Mismo método que HybridEvaluator._compute_agreement: pesos geométricos
        [1.0, 0.5, 0.25, 0.125, 0.0625] sobre top-5 features.

        Esto es coherente con la métrica que el evaluador usa para exp_agreement
        y evita que agentes con desacuerdo SHAP/LIME informado (fidelity alta
        pero features distintas) sean penalizados con Jaccard=0.
        """
        explainer_names = list(per_explainer.keys())

        if len(explainer_names) < 2:
            return [1.0] * n

        _GEOM_WEIGHTS = [1.0, 0.5, 0.25, 0.125, 0.0625]

        def rank_weights(v, k=5):
            indices = np.argsort(np.abs(v))[::-1][:k]
            return {int(idx): _GEOM_WEIGHTS[rank]
                    for rank, idx in enumerate(indices)
                    if rank < len(_GEOM_WEIGHTS)}

        agreement = []
        for i in range(n):
            weighted_vecs = []
            for name in explainer_names:
                v = self._get_explanation_vector(responses[i], name)
                if v is not None and np.linalg.norm(v) > 1e-8:
                    weighted_vecs.append(rank_weights(v))

            if len(weighted_vecs) < 2:
                agreement.append(0.5)
                continue

            pairs = scores_sum = 0
            for a in range(len(weighted_vecs)):
                for b in range(a + 1, len(weighted_vecs)):
                    pairs += 1
                    all_feats   = set(weighted_vecs[a]) | set(weighted_vecs[b])
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
    # Utils
    # ======================================================

    def _get_explanation_vector(self, response, explainer_name):
        for exp in response.get("explanations", []):
            if (exp.get("explainer") == explainer_name
                    and "details" in exp
                    and "values" in exp["details"]):
                v = np.asarray(exp["details"]["values"], dtype=float)
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                if np.linalg.norm(v) > 1e-8:
                    return v
        return None

    def _top3_indices(self, vector):
        if vector is None or len(vector) == 0:
            return []
        return np.argsort(np.abs(vector))[::-1][:3].tolist()