import numpy as np


class ConflictResolver:
    """
    Resolución de conflictos multi-criterio y explanation-aware.

    Usa:
    - score global
    - confianza del modelo
    - calidad explicativa
    - consenso inter-agente
    - estabilidad temporal
    - fidelidad modelo-explicación
    """

    def __init__(
        self,
        low_q=0.25,
        high_q=0.75,
        min_exp_quality=0.3,
        min_stability=0.4,
        min_fidelity=0.4,
        consensus_stop=0.85
    ):
        self.low_q = low_q
        self.high_q = high_q
        self.min_exp_quality = min_exp_quality
        self.min_stability = min_stability
        self.min_fidelity = min_fidelity
        self.consensus_stop = consensus_stop

    def resolve(self, evaluation):
        scores = np.asarray(evaluation["scores"], dtype=float)
        conf   = np.asarray(evaluation["components"]["conf"], dtype=float)
        exp_q  = np.asarray(evaluation["components"]["exp"], dtype=float)

        exp_d = evaluation["components"].get("exp_detail", {})

        consensus = np.asarray(
            exp_d.get("consensus", exp_q), dtype=float
        )
        stability = np.asarray(
            exp_d.get("stability", np.full_like(exp_q, 0.5)), dtype=float
        )
        fidelity = np.asarray(
            exp_d.get("fidelity", np.full_like(exp_q, 0.5)), dtype=float
        )

        low  = np.quantile(scores, self.low_q)
        high = np.quantile(scores, self.high_q)

        decisions = []

        for s, c, e, stab, fid in zip(scores, conf, exp_q, stability, fidelity):

            # 🔴 Riesgo crítico: modelo confiado pero explica mal
            if c > 0.7 and (e < self.min_exp_quality or fid < self.min_fidelity):
                decisions.append("force_adjust")
                continue

            # 🔴 Explicaciones inestables
            if stab < self.min_stability:
                decisions.append("adjust")
                continue

            # 🟡 Baja fidelidad explicativa
            if fid < self.min_fidelity:
                decisions.append("soft_adjust")
                continue

            # ⚪ Decisión basada en score global
            if s >= high:
                decisions.append("keep")
            elif s <= low:
                decisions.append("adjust")
            else:
                decisions.append("soft_adjust")

        # 🔚 Criterio de parada por consenso explicativo estable
        stop = (
            np.mean(consensus) >= self.consensus_stop and
            np.mean(stability) >= self.min_stability and
            all(d in {"keep", "soft_adjust"} for d in decisions)
        )

        return {
            "decisions": decisions,
            "stop": bool(stop),
            "diagnostics": {
                "mean_consensus": float(np.mean(consensus)),
                "mean_stability": float(np.mean(stability)),
                "mean_fidelity": float(np.mean(fidelity))
            }
        }