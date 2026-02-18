# aggregator_agent.py
import asyncio
from collections import defaultdict
from agents.message import Message
from agents.evaluation.hybrid_evaluator import HybridEvaluator
from agents.evaluation.conflict_resolver import ConflictResolver
from agents.evaluation.feedback_builder import FeedbackBuilder


class AggregatorAgent:

    def __init__(self, classifier_ids, max_iterations=3,
                 alpha=0.5, beta=0.2, gamma=0.3):

        self.classifier_ids = classifier_ids
        self.max_iterations = max_iterations

        self.evaluator = HybridEvaluator(alpha, beta, gamma)
        self.resolver = ConflictResolver()
        self.feedback_builder = FeedbackBuilder()

        self.inbox = asyncio.Queue()
        self.current_iteration = 0

        self.instance = None
        self.buffer = defaultdict(list)
        self.global_history = []

    async def run(self, queues, instance):
        self.instance = instance
        print("[Aggregator] Inicio")

        while self.current_iteration < self.max_iterations:
            print(f"\n[Aggregator] Iteración {self.current_iteration}")

            # -------- Solicitud de predicciones --------
            for cid in self.classifier_ids:
                await queues[cid].put(
                    Message(
                        sender="aggregator",
                        body={
                            "action": "classify",
                            "iteration": self.current_iteration,
                            "instance": self.instance
                        }
                    )
                )

            # -------- Recoger respuestas --------
            responses = [
                (await self.inbox.get()).body
                for _ in self.classifier_ids
            ]

            self.buffer[self.current_iteration] = responses
            self.global_history.append(responses)

            # -------- Evaluación --------
            evaluation = self.evaluator.evaluate(responses)

            # -------- Resolución de conflictos --------
            decisions = self.resolver.resolve(evaluation)

            print(f"[Aggregator] Majority = {evaluation['majority_prediction']}")
            print(f"[Aggregator] Scores = {evaluation['scores']}")
            print(f"[Aggregator] Decisions = {decisions}")

            # -------- Feedback RL --------
            for idx, (cid, decision) in enumerate(zip(self.classifier_ids, decisions)):
                feedback = self.feedback_builder.build(
                    agent_id=cid,
                    decision=decision,
                    evaluation=evaluation,
                    explanations=responses[idx].get("explanations", []),
                    idx=idx
                )

                await queues[cid].put(
                    Message(sender="aggregator", body=feedback)
                )

            self.current_iteration += 1

        print("\n[Aggregator] Finalizado")
