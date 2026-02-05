# agents/aggregator_agent.py

import json
import asyncio
from collections import defaultdict
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message


class AggregatorAgent(Agent):
    """
    Agente agregador que combina decisiones de múltiples agentes clasificadores.
    """

    def __init__(self, jid, password, classifier_agents, strategy="weighted_voting", **kwargs):
        super().__init__(jid, password, **kwargs)
        self.classifier_agents = classifier_agents
        self.strategy = strategy

    async def setup(self):
        print(f"[{self.jid}] Agente agregador listo.")

    async def classify(self, X, instance_id=0, timeout=10):
        """
        Lanza una petición de clasificación a todos los agentes
        y agrega las respuestas.
        """
        behaviour = self.AggregationBehaviour(X, instance_id, timeout)
        self.add_behaviour(behaviour)
        return await behaviour.result()

    class AggregationBehaviour(OneShotBehaviour):
        def __init__(self, X, instance_id, timeout):
            super().__init__()
            self.X = X
            self.instance_id = instance_id
            self.timeout = timeout
            self._result = None

        async def run(self):
            # 1️⃣ Enviar peticiones
            for agent_jid in self.agent.classifier_agents:
                msg = Message(to=agent_jid)
                msg.set_metadata("performative", "request")
                msg.body = json.dumps({
                    "action": "classify",
                    "data": self.X,
                    "instance_id": self.instance_id
                })
                await self.send(msg)

            # 2️⃣ Recibir respuestas
            responses = []
            for _ in range(len(self.agent.classifier_agents)):
                reply = await self.receive(timeout=self.timeout)
                if reply is not None:
                    responses.append(json.loads(reply.body))

            # 3️⃣ Agregación
            final_prediction, confidence = self.aggregate(responses)

            # 4️⃣ Meta-explicación
            meta_explanation = {
                "strategy": self.agent.strategy,
                "individual_decisions": responses
            }

            self._result = {
                "final_prediction": final_prediction,
                "confidence": confidence,
                "strategy": self.agent.strategy,
                "agents_used": [r["agent"] for r in responses],
                "meta_explanation": meta_explanation
            }

        def aggregate(self, responses):
            """
            Estrategias de agregación.
            """
            votes = defaultdict(float)

            for r in responses:
                pred = r["prediction"]
                conf = r.get("confidence", 1.0)
                weight = conf if self.agent.strategy == "weighted_voting" else 1.0
                votes[pred] += weight

            final_pred = max(votes, key=votes.get)
            total_weight = sum(votes.values())
            confidence = votes[final_pred] / total_weight if total_weight > 0 else None

            return final_pred, confidence

        async def result(self):
            while self._result is None:
                await asyncio.sleep(0.1)
            return self._result
