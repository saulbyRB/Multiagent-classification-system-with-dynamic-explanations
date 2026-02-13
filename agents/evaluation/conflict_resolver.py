import numpy as np


class ConflictResolver:
    """
    Decide estrategia según score, estabilidad y coherencia
    """

    def __init__(
        self,
        low_q=0.25,
        high_q=0.75,
        min_exp_quality=0.3
    ):
        self.low_q = low_q
        self.high_q = high_q
        self.min_exp_quality = min_exp_quality

    def resolve(self, evaluation):
        scores = np.array(evaluation["scores"])
        exp_q  = np.array(evaluation["components"]["exp"])
        conf_q = np.array(evaluation["components"]["conf"])

        low  = np.quantile(scores, self.low_q)
        high = np.quantile(scores, self.high_q)

        resolutions = []

        for s, e, c in zip(scores, exp_q, conf_q):

            # Caso peligroso: muy seguro pero explica mal
            if c > 0.7 and e < self.min_exp_quality:
                resolutions.append("force_adjust")
                continue

            if s >= high:
                resolutions.append("keep")
            elif s <= low:
                resolutions.append("adjust")
            else:
                resolutions.append("soft_adjust")

        return resolutions
