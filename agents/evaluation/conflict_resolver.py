import numpy as np


class ConflictResolver:
    """
    Resolución de conflictos multi-criterio y explicación-aware.
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
        scores = np.array(evaluation["scores"])
        conf   = np.array(evaluation["components"]["conf"])
        exp    = np.array(evaluation["components"]["exp"])

        exp_d = evaluation["components"].get("exp_detail", {})

        consensus = np.array(exp_d.get("consensus", exp))
        stability = np.array(exp_d.get("stability", np.ones_like(exp)))
        fidelity  = np.array(exp_d.get("fidelity", np.ones_like(exp)))

        low  = np.quantile(scores, self.low_q)
        high = np.quantile(scores, self.high_q)

        resolutions = []

        for s, c, e, stab, fid in zip(scores, conf, exp, stability, fidelity):

            # 🟥 Caso peligroso
            if c > 0.7 and (e < self.min_exp_quality or fid < self.min_fidelity):
                resolutions.append("force_adjust")
                continue

            # 🟥 Inestabilidad explicativa
            if stab < self.min_stability:
                resolutions.append("adjust")
                continue

            # 🟡 Baja fidelidad pero buen score
            if fid < self.min_fidelity:
                resolutions.append("soft_adjust")
                continue

            # Score-based (fallback)
            if s >= high:
                resolutions.append("keep")
            elif s <= low:
                resolutions.append("adjust")
            else:
                resolutions.append("soft_adjust")

        # 🔚 Criterio de parada temprana
        stop = (
            np.mean(consensus) > self.consensus_stop
            and np.mean(stability) > self.min_stability
            and all(r in {"keep", "soft_adjust"} for r in resolutions)
        )

        return resolutions, stop