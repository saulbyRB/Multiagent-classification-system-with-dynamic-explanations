import numpy as np
from collections import deque


class ConflictResolver:

    def __init__(
        self,
        low_q=0.25,
        high_q=0.75,
        min_exp_quality=0.50,
        min_stability=0.75,
        min_fidelity=0.5,
        consensus_stop=0.65,
        satisfaction_window=5,      # iteraciones estables para votar parada
        min_accuracy_stop=0.85      # accuracy mínima para poder parar
    ):
        self.low_q              = low_q
        self.high_q             = high_q
        self.min_exp_quality    = min_exp_quality
        self.min_stability      = min_stability
        self.min_fidelity       = min_fidelity
        self.consensus_stop     = consensus_stop
        self.satisfaction_window = satisfaction_window
        self.min_accuracy_stop  = min_accuracy_stop

        # Historial de satisfacción por agente (idx → deque)
        self._satisfaction_history = {}
        # Historial de scores por agente para detectar plateau
        self._score_history = {}

    def resolve(self, evaluation):
        scores   = np.asarray(evaluation["scores"],                  dtype=float)
        conf     = np.asarray(evaluation["components"]["conf"],      dtype=float)
        exp_q    = np.asarray(evaluation["components"]["exp"],       dtype=float)
        accuracy = np.asarray(evaluation["components"].get("acc",
                              np.full_like(scores, 0.9)),            dtype=float)

        exp_d     = evaluation["components"].get("exp_detail", {})
        consensus = np.asarray(exp_d.get("consensus", exp_q),                         dtype=float)
        stability = np.asarray(exp_d.get("stability", np.full_like(exp_q, 0.5)),      dtype=float)
        fidelity  = np.asarray(exp_d.get("fidelity",  np.full_like(exp_q, 0.5)),      dtype=float)
        quality   = np.asarray(exp_d.get("quality",   exp_q),                         dtype=float)

        n = len(scores)

        # Inicializar historiales si es la primera vez
        for i in range(n):
            if i not in self._satisfaction_history:
                self._satisfaction_history[i] = deque(maxlen=self.satisfaction_window)
                self._score_history[i]        = deque(maxlen=self.satisfaction_window)

        low  = np.quantile(scores, self.low_q)
        high = np.quantile(scores, self.high_q)

        decisions       = []
        agent_satisfied = []
        agent_votes_stop = []  # voto individual de parada

        for i, (s, c, e, stab, fid, q, acc) in enumerate(
            zip(scores, conf, exp_q, stability, fidelity, quality, accuracy)
        ):
            # ── Actualizar historial ───────────────────────────────────────
            self._score_history[i].append(s)

            # ── Decisión de ajuste ────────────────────────────────────────
            if c > 0.7 and e < self.min_exp_quality:
                decisions.append("force_adjust")
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            if self.min_fidelity > 0 and c > 0.7 and fid < self.min_fidelity:
                decisions.append("force_adjust")
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            if stab < self.min_stability:
                decisions.append("adjust")
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            if s >= high:
                decisions.append("keep")
            elif s <= low:
                decisions.append("adjust")
            else:
                decisions.append("soft_adjust")

            # ── Satisfacción individual ───────────────────────────────────
            satisfied = (
                q    >= self.min_exp_quality and
                stab >= self.min_stability   and
                fid  >= self.min_fidelity    and
                acc  >= self.min_accuracy_stop
            )
            agent_satisfied.append(satisfied)
            self._satisfaction_history[i].append(satisfied)

            # ── Voto de parada: satisfecho en toda la ventana ─────────────
            # El agente vota parar si ha estado satisfecho las últimas
            # N iteraciones Y su score es estable (std bajo)
            history = list(self._satisfaction_history[i])
            scores_window = list(self._score_history[i])

            window_full      = len(history) >= self.satisfaction_window
            all_satisfied    = window_full and all(history)
            score_stable     = (window_full and
                               np.std(scores_window) < 0.08)

            votes_stop = all_satisfied and score_stable
            agent_votes_stop.append(votes_stop)

        # ── Criterio de parada global ──────────────────────────────────────
        # Todos los agentes votan parar (consenso unánime de satisfacción)
        all_vote_stop     = all(agent_votes_stop)
        no_hard_adjust    = all(d in {"keep", "soft_adjust"} for d in decisions)
        global_consensus  = float(np.mean(consensus)) >= self.consensus_stop

        stop = all_vote_stop and no_hard_adjust and global_consensus

        return {
            "decisions": decisions,
            "stop": bool(stop),
            "diagnostics": {
                "mean_consensus":          float(np.mean(consensus)),
                "mean_stability":          float(np.mean(stability)),
                "mean_fidelity":           float(np.mean(fidelity)),
                "agent_satisfied":         agent_satisfied,
                "agent_votes_stop":        agent_votes_stop,
                "all_vote_stop":           all_vote_stop,
                "global_consensus_ok":     global_consensus,
            }
        }