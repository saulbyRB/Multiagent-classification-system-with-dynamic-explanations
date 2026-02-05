# agents/classifier_agent.py

import json
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message


class ClassifierAgent(Agent):
    """
    Agente clasificador que alberga un modelo de ML y un explainer.
    """

    def __init__(self, jid, password, model, explainer, **kwargs):
        super().__init__(jid, password, **kwargs)
        self.model = model
        self.explainer = explainer

    async def setup(self):
        """
        Inicializa el comportamiento del agente.
        """
        self.add_behaviour(self.ClassificationBehaviour())
        print(f"[{self.jid}] Agente clasificador listo.")

    class ClassificationBehaviour(CyclicBehaviour):
        """
        Comportamiento cíclico para atender peticiones de clasificación.
        """

        async def run(self):
            msg = await self.receive(timeout=10)
            if msg is None:
                return

            try:
                content = json.loads(msg.body)
                action = content.get("action")

                if action != "classify":
                    return

                X = content.get("data")
                instance_id = content.get("instance_id", 0)

                # 1️⃣ Predicción
                y_pred = int(self.agent.model.predict(X)[instance_id])
                confidence = self.agent.model.get_confidence(X)
                confidence = float(confidence[instance_id]) if confidence is not None else None

                # 2️⃣ Explicación dinámica
                explanation = self.agent.explainer.explain(
                    model=self.agent.model,
                    X=X,
                    instance_id=instance_id
                )

                # 3️⃣ Respuesta
                response = {
                    "agent": str(self.agent.jid),
                    "prediction": y_pred,
                    "confidence": confidence,
                    "explanation": explanation
                }

                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", "inform")
                reply.body = json.dumps(response)

                await self.send(reply)

            except Exception as e:
                error_msg = Message(to=str(msg.sender))
                error_msg.set_metadata("performative", "failure")
                error_msg.body = json.dumps({"error": str(e)})
                await self.send(error_msg)
