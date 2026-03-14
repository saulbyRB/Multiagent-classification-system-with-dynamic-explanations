import asyncio
import numpy as np
from agents.message import Message
from agents.evaluation.hybrid_evaluator import HybridEvaluator
from agents.evaluation.conflict_resolver import ConflictResolver
from agents.evaluation.feedback_builder import FeedbackBuilder
from visualization.logs import log


class AggregatorAgent:

    def __init__(self, classifier_ids, max_iterations=50,
                 alpha=0.5, beta=0.2, gamma=0.3, background_data=None):

        self.classifier_ids = classifier_ids
        self.max_iterations = max_iterations

        self.evaluator        = HybridEvaluator(background_data=background_data)
        self.resolver         = ConflictResolver()
        self.feedback_builder = FeedbackBuilder()

        self.inbox             = asyncio.Queue()
        self.current_iteration = 0
        self.instance          = None
        self.global_history    = []
        self._score_history    = []

    async def run(self, queues, instance):
        self.instance = instance
        print("[Aggregator] Inicio")

        while self.current_iteration < self.max_iterations:
            print(f"\n[Aggregator] Iteración {self.current_iteration}")

            # ── Solicitud de predicciones ──────────────────────────────────
            for cid in self.classifier_ids:
                await queues[cid].put(Message(
                    sender="aggregator",
                    body={
                        "action":    "classify",
                        "iteration": self.current_iteration,
                        "instance":  self.instance
                    }
                ))

            # ── Recoger respuestas ─────────────────────────────────────────
            responses = [
                (await self.inbox.get()).body
                for _ in self.classifier_ids
            ]
            log(f"Recibidas {len(responses)} respuestas", "AGGREGATOR")

            # ── Evaluación ─────────────────────────────────────────────────
            evaluation = self.evaluator.evaluate(responses)
            scores     = evaluation["scores"]
            self._score_history.append(scores)

            # ── Tendencia individual por agente ────────────────────────────
            if len(self._score_history) >= 2:
                trends = [
                    float(c - p)
                    for c, p in zip(self._score_history[-1], self._score_history[-2])
                ]
            else:
                trends = [0.0] * len(self.classifier_ids)

            global_trend = float(np.mean(trends))

            # ── Resolución ─────────────────────────────────────────────────
            resolution  = self.resolver.resolve(evaluation)
            decisions   = resolution["decisions"]
            stop        = resolution["stop"]
            diagnostics = resolution["diagnostics"]

            print(f"[Aggregator] Majority   = {evaluation['majority_prediction']}")
            print(f"[Aggregator] Scores     = {[round(s,3) for s in scores]}")
            print(f"[Aggregator] Trends     = {[round(t,3) for t in trends]}")
            print(f"[Aggregator] Decisions  = {decisions}")
            print(f"[Aggregator] VotosStop  = {diagnostics['agent_votes_stop']}")
            print(f"[Aggregator] Consensus  = {diagnostics['mean_consensus']:.3f}")
            print(f"[Aggregator] Stop       = {stop}")

            # ── Historial ──────────────────────────────────────────────────
            self.global_history.append({
                "iteration":   self.current_iteration,
                "responses":   responses,
                "evaluation":  evaluation,
                "decisions":   decisions,
                "trends":      trends,
                "diagnostics": diagnostics,
                "stop":        stop
            })

            if stop:
                print(f"\n[Aggregator] ✓ Consenso unánime en iter "
                      f"{self.current_iteration} → terminación anticipada")
                break

            # ── Identificar mentor ─────────────────────────────────────────
            log("Buscando mentor", "AGGREGATOR")
            mentor_vector, mentor_ids, is_dissent = self.feedback_builder.find_mentor(
                responses, evaluation
            )
            if mentor_ids:
                log(f"Mentor(s) encontrado(s): {mentor_ids}", "AGGREGATOR")
            else:
                log("Sin mentor esta iteración (ningún agente cumple umbrales)",
                    "AGGREGATOR")

            # ── Feedback ───────────────────────────────────────────────────
            log("Calculando feedback global", "AGGREGATOR")
            for idx, (cid, decision) in enumerate(
                zip(self.classifier_ids, decisions)
            ):
                feedback = self.feedback_builder.build_with_mentor(
                    agent_id=cid,
                    decision=decision,
                    evaluation=evaluation,
                    explanations=responses[idx].get("explanations", []),
                    idx=idx,
                    mentor_vector=mentor_vector,
                    mentor_ids=mentor_ids,
                    is_dissent_mentor=is_dissent
                )

                feedback["action"]       = "feedback"
                feedback["iteration"]    = self.current_iteration
                feedback["trend"]        = trends[idx]
                feedback["global_trend"] = global_trend

                # Pasar también la instancia y la predicción mayoritaria
                # para que los modelos puedan hacer fine-tune dirigido
                feedback["instance"]      = self.instance
                feedback["target_pred"]   = evaluation["majority_prediction"]

                await queues[cid].put(
                    Message(sender="aggregator", body=feedback)
                )

            log(f"Iteración {self.current_iteration} completada", "AGGREGATOR")
            self.current_iteration += 1

        # ── Shutdown ───────────────────────────────────────────────────────
        print("\n[Aggregator] Finalizado → enviando shutdown")
        for cid in self.classifier_ids:
            await queues[cid].put(
                Message(sender="aggregator", body={"action": "shutdown"})
            )