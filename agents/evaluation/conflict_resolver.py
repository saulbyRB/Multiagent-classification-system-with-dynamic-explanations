import numpy as np
from collections import deque


class ConflictResolver:

    def __init__(
        self,
        low_q=0.25,
        high_q=0.75,
        min_exp_quality=0.35,
        min_stability=0.60,
        min_fidelity=0.0,
        min_agreement=0.5,
        consensus_stop=0.45,
        satisfaction_window=3,
        min_accuracy_stop=0.70,
        warmup_iterations=3,
        max_disagreement_iters=8,   # iters consecutivas discrepando → force_adjust
    ):
        self.low_q                  = low_q
        self.high_q                 = high_q
        self.min_exp_quality        = min_exp_quality
        self.min_stability          = min_stability
        self.min_fidelity           = min_fidelity
        self.min_agreement          = min_agreement
        self.consensus_stop         = consensus_stop
        self.satisfaction_window    = satisfaction_window
        self.min_accuracy_stop      = min_accuracy_stop
        self.warmup_iterations      = warmup_iterations
        self.max_disagreement_iters = max_disagreement_iters

        self._satisfaction_history  = {}
        self._score_history         = {}
        self._disagreement_streak   = {}   # contador de iters consecutivas discrepando
        self._iteration             = 0

    def resolve(self, evaluation):
        scores   = np.asarray(evaluation["scores"],             dtype=float)
        conf     = np.asarray(evaluation["components"]["conf"], dtype=float)
        exp_q    = np.asarray(evaluation["components"]["exp"],  dtype=float)
        accuracy = np.asarray(
            evaluation["components"].get("acc", np.full_like(scores, 0.9)),
            dtype=float
        )
        preds    = evaluation.get("majority_prediction", None)
        # Predicciones individuales para detectar discrepancia
        ind_preds = evaluation.get("individual_predictions", [None] * len(scores))

        exp_d     = evaluation["components"].get("exp_detail", {})
        consensus = np.asarray(
            exp_d.get("consensus", exp_q), dtype=float)
        stability = np.asarray(
            exp_d.get("stability", np.full_like(exp_q, 0.5)), dtype=float)
        fidelity  = np.asarray(
            exp_d.get("fidelity",  np.full_like(exp_q, 0.5)), dtype=float)
        quality   = np.asarray(
            exp_d.get("quality",   exp_q), dtype=float)
        agreement = np.asarray(
            exp_d.get("agreement", np.full_like(exp_q, 1.0)), dtype=float)

        majority = evaluation.get("majority_prediction", None)

        n = len(scores)
        for i in range(n):
            if i not in self._satisfaction_history:
                self._satisfaction_history[i] = deque(maxlen=self.satisfaction_window)
                self._score_history[i]        = deque(maxlen=self.satisfaction_window)
                self._disagreement_streak[i]  = 0

        # ── Actualizar streaks de discrepancia ─────────────────────────────
        # ind_preds viene del evaluador; si no está disponible usamos S_pred < 0.5
        # como proxy (S_pred ponderado por confianza: < 0.5 → discrepa).
        S_pred = np.asarray(evaluation["components"].get("pred", [1.0] * n), dtype=float)
        for i in range(n):
            if S_pred[i] < 0.5:
                self._disagreement_streak[i] += 1
            else:
                self._disagreement_streak[i] = 0

        low  = np.quantile(scores, self.low_q)
        high = np.quantile(scores, self.high_q)

        decisions        = []
        agent_satisfied  = []
        agent_votes_stop = []

        for i, (s, c, e, stab, fid, q, acc, agr) in enumerate(
            zip(scores, conf, exp_q, stability, fidelity, quality, accuracy, agreement)
        ):
            self._score_history[i].append(s)

            # ── Regla 0: discrepancia persistente ─────────────────────────
            # Si el agente lleva >= max_disagreement_iters iteraciones
            # consecutivas prediciendo diferente a la mayoría, force_adjust
            # independientemente de sus otras métricas.
            # Solo se aplica fuera del warmup para dar tiempo a estabilizarse.
            in_warmup = self._iteration < self.warmup_iterations
            streak    = self._disagreement_streak[i]

            # Un agente de alta calidad que discrepa persistentemente
            # NO recibe force_adjust: su disenso puede ser informativo,
            # especialmente en instancias frontera donde la mayoría es
            # poco fiable. Solo se penaliza si además tiene baja calidad.
            high_quality_dissenter = (
                acc >= self.min_accuracy_stop and
                q   >= self.min_exp_quality   and
                fid >= 0.70
            )
            if not in_warmup and streak >= self.max_disagreement_iters:
                if high_quality_dissenter:
                    # Disenso informado: soft_adjust como máximo
                    decisions.append("soft_adjust")
                else:
                    decisions.append("force_adjust")
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            # ── Regla 1: alta confianza + baja calidad explicativa ─────────
            if c > 0.7 and e < self.min_exp_quality:
                if acc < self.min_accuracy_stop:
                    decisions.append("force_adjust")
                else:
                    decisions.append("adjust")
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            # ── Regla 2: alta confianza + baja fidelity ───────────────────
            if self.min_fidelity > 0 and c > 0.7 and fid < self.min_fidelity:
                if acc < self.min_accuracy_stop:
                    decisions.append("force_adjust")
                else:
                    decisions.append("adjust")
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            # ── Regla 3: baja estabilidad ──────────────────────────────────
            if stab < self.min_stability:
                decisions.append("adjust")
                agent_satisfied.append(False)
                self._satisfaction_history[i].append(False)
                agent_votes_stop.append(False)
                continue

            # ── Decisión relativa al grupo ─────────────────────────────────
            if s >= high:
                decisions.append("keep")
            elif s <= low:
                if q >= self.min_exp_quality and stab >= self.min_stability and acc >= self.min_accuracy_stop:
                    decisions.append("soft_adjust")
                else:
                    decisions.append("adjust")
            else:
                decisions.append("soft_adjust")

            # ── Satisfacción individual ────────────────────────────────────
            satisfied = (
                q    >= self.min_exp_quality   and
                stab >= self.min_stability     and
                fid  >= self.min_fidelity      and
                acc  >= self.min_accuracy_stop and
                agr  >= self.min_agreement     and
                streak == 0                        # no satisfecho si discrepa
            )
            agent_satisfied.append(satisfied)
            self._satisfaction_history[i].append(satisfied)

            # ── Voto de parada ─────────────────────────────────────────────
            history       = list(self._satisfaction_history[i])
            scores_window = list(self._score_history[i])

            window_full   = len(history) >= self.satisfaction_window
            all_satisfied = window_full and all(history)
            score_stable  = window_full and np.std(scores_window) < 0.08

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
                "mean_consensus":        float(np.mean(consensus)),
                "mean_stability":        float(np.mean(stability)),
                "mean_fidelity":         float(np.mean(fidelity)),
                "mean_agreement":        float(np.mean(agreement)),
                "agent_satisfied":       agent_satisfied,
                "agent_votes_stop":      agent_votes_stop,
                "all_vote_stop":         all_vote_stop,
                "global_consensus_ok":   global_consensus,
                "disagreement_streaks":  [self._disagreement_streak[i] for i in range(n)],
                "iteration":             self._iteration,
            }
        }