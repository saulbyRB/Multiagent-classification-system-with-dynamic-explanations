import numpy as np


class FeedbackBuilder:

    def build(self, agent_id, decision, evaluation, explanations=None, idx=0):
        scores      = evaluation["scores"]
        n           = len(scores)
        my_score    = scores[idx]
        peer_scores = [s for i, s in enumerate(scores) if i != idx]

        exp_detail  = evaluation["components"].get("exp_detail", {})
        quality_arr = exp_detail.get("quality",   [0.5] * n)
        stab_arr    = exp_detail.get("stability", [0.5] * n)

        # Posición relativa del agente respecto al grupo
        # +1.0 = mejor del grupo, -1.0 = peor del grupo
        if len(peer_scores) > 0:
            mean_peer   = float(np.mean(peer_scores))
            relative_pos = float(np.tanh((my_score - mean_peer) * 5))
        else:
            mean_peer    = my_score
            relative_pos = 0.0

        # Si todos los peers también están mal, el problema es estructural
        all_peers_struggling = (
            len(peer_scores) > 0 and
            all(s < 0.5 for s in peer_scores)
        )

        # Presión de grupo: cuánto diverge este agente de la media
        # positivo → estoy por encima → puedo ser más conservador
        # negativo → estoy por debajo → debo ajustar más
        group_pressure = float(my_score - mean_peer)

        return {
            "strategy":    decision,
            "agent_id":    agent_id,
            "evaluation": {
                "reward":                   my_score,
                "prediction_alignment":     evaluation["components"]["pred"][idx],
                "confidence":               evaluation["components"]["conf"][idx],
                "explanation_similarity":   evaluation["components"]["exp"][idx],
                "exp_quality":              quality_arr[idx],
                "exp_consensus":            exp_detail.get("consensus", [0.5]*n)[idx],
                "exp_stability":            stab_arr[idx],
                "num_explainers":           len(explanations) if explanations else 0,
                "global_score":             evaluation["global_score"],
            },
            # Señales de grupo — el agente las usa para ajustarse
            # en relación a sus peers, no solo a sí mismo
            "group_signals": {
                "peer_scores":           peer_scores,
                "mean_peer_score":       mean_peer,
                "relative_position":     relative_pos,   # mi posición vs el grupo
                "group_pressure":        group_pressure,  # cuánto divergo
                "all_peers_struggling":  all_peers_struggling,
                "group_quality_mean":    float(np.mean(quality_arr)),
                "group_stability_mean":  float(np.mean(stab_arr)),
            }
        }