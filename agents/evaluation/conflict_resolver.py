# conflict_resolver.py
import numpy as np
from collections import deque


class ConflictResolver:

    def __init__(
        self,
        low_q=0.25,
        high_q=0.75,
        min_exp_quality=0.45,      # ← bajado de 0.50
        min_stability=0.75,
        min_fidelity=0.0,
        consensus_stop=0.65,
        satisfaction_window=4,     # ← bajado de 5
        min_accuracy_stop=0.85,
        warmup_iterations=5     # no votar parada durante calentamiento
        ):
        self.low_q               = low_q
        self.high_q              = high_q
        self.min_exp_quality     = min_exp_quality
        self.min_stability       = min_stability
        self.min_fidelity        = min_fidelity
        self.consensus_stop      = consensus_stop
        self.satisfaction_window = satisfaction_window
        self.min_accuracy_stop   = min_accuracy_stop
        self.warmup_iterations   = warmup_iterations

        self._satisfaction_history = {}
        self._score_history        = {}
        self._iteration            = 0

    def resolve(self, evaluation):
        scores   = np.asarray(evaluation["scores"],             dtype=float)
        conf     = np.asarray(evaluation["components"]["conf"], dtype=float)
        exp_q    = np.asarray(evaluation["components"]["exp"],  dtype=float)
        accuracy = np.asarray(
            evaluation["components"].get("acc", np.full_like(scores, 0.9)),
            dtype=float
        )

        exp_d     = evaluation["components"].get("exp_detail", {})
        consensus = np.asarray(
            exp_d.get("consensus", exp_q), dtype=float)
        stability = np.asarray(
            exp_d.get("stability", np.full_like(exp_q, 0.5)), dtype=float)
        fidelity  = np.asarray(
            exp_d.get("fidelity",  np.full_like(exp_q, 0.5)), dtype=float)
        quality   = np.asarray(
            exp_d.get("quality",   exp_q), dtype=float)

        n = len(scores)
        for i in range(n):
            if i not in self._satisfaction_history:
                self._satisfaction_history[i] = deque(
                    maxlen=self.satisfaction_window)
                self._score_history[i] = deque(
                    maxlen=self.satisfaction_window)

        low  = np.quantile(scores, self.low_q)
        high = np.quantile(scores, self.high_q)

        decisions        = []
        agent_satisfied  = []
        agent_votes_stop = []

        for i, (s, c, e, stab, fid, q, acc) in enumerate(
            zip(scores, conf, exp_q, stability, fidelity, quality, accuracy)
        ):
            self._score_history[i].append(s)

            # ── Decisión de ajuste ─────────────────────────────────────────
            # force_adjust solo si accuracy también es baja — si el modelo
            # ya clasifica bien, un adjust es suficiente para mejorar
            # las explicaciones sin desestabilizarlo
            if c > 0.7 and e < self.min_exp_quality:
                if acc < self.min_accuracy_stop:
                    decisions.append("force_adjust")
                else:
                    decisions.append("adjust")   # acc ok → ajuste suave
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            if self.min_fidelity > 0 and c > 0.7 and fid < self.min_fidelity:
                if acc < self.min_accuracy_stop:
                    decisions.append("force_adjust")
                else:
                    decisions.append("adjust")
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

            # Decisión relativa al grupo
            if s >= high:
                decisions.append("keep")
            elif s <= low:
                # Solo adjust si realmente tiene problemas individuales
                if q >= self.min_exp_quality and stab >= self.min_stability and acc >= self.min_accuracy_stop:
                    decisions.append("soft_adjust")  # bueno pero el peor → suave
                else:
                    decisions.append("adjust")       # malo → ajuste real
            else:
                decisions.append("soft_adjust")

            # ── Satisfacción individual ────────────────────────────────────
            satisfied = (
                q    >= self.min_exp_quality and
                stab >= self.min_stability   and
                fid  >= self.min_fidelity    and
                acc  >= self.min_accuracy_stop
            )
            agent_satisfied.append(satisfied)
            self._satisfaction_history[i].append(satisfied)

            # ── Voto de parada ─────────────────────────────────────────────
            history       = list(self._satisfaction_history[i])
            scores_window = list(self._score_history[i])

            in_warmup    = self._iteration < self.warmup_iterations
            window_full  = len(history) >= self.satisfaction_window
            all_satisfied = window_full and all(history)
            score_stable  = window_full and np.std(scores_window) < 0.08

            # Durante calentamiento nunca votar parada
            votes_stop = (not in_warmup) and all_satisfied and score_stable
            agent_votes_stop.append(votes_stop)

        # ── Criterio de parada global ──────────────────────────────────────
        all_vote_stop    = all(agent_votes_stop)
        no_hard_adjust   = all(d in {"keep", "soft_adjust"} for d in decisions)
        global_consensus = float(np.mean(consensus)) >= self.consensus_stop

        stop = all_vote_stop and no_hard_adjust and global_consensus

        self._iteration += 1

        return {
            "decisions": decisions,
            "stop":      bool(stop),
            "diagnostics": {
                "mean_consensus":      float(np.mean(consensus)),
                "mean_stability":      float(np.mean(stability)),
                "mean_fidelity":       float(np.mean(fidelity)),
                "agent_satisfied":     agent_satisfied,
                "agent_votes_stop":    agent_votes_stop,
                "all_vote_stop":       all_vote_stop,
                "global_consensus_ok": global_consensus,
                "iteration":           self._iteration,
            }
        }