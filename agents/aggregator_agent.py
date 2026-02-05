# agents/classifier_agent.py

import json
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from sklearn.model_selection import train_test_split

class ClassifierAgent(Agent):
    """
    Agente clasificador que alberga un modelo de ML y un explainer.
    Se entrena automáticamente con un dataset de DatasetRegistry antes de atender mensajes.
    """

    def __init__(self, jid, password, model, explainer, dataset_id=None, registry=None, test_size=0.2, **kwargs):
        super().__init__(jid, password, **kwargs)
        self.model = model
        self.explainer = explainer
        self.dataset_id = dataset_id
        self.registry = registry
        self.test_size = test_size

        self.X_test = None
        self.y_test = None

    async def setup(self):
        # 1️⃣ Behaviour de entrenamiento
        self.add_behaviour(self.TrainingBehaviour())

        # 2️⃣ Behaviour cíclico de clasificación
        self.add_behaviour(self.ClassificationBehaviour())
        print(f"[{self.jid}] Agente clasificador listo.")

    # ==================== Training Behaviour ====================
    class TrainingBehaviour(OneShotBehaviour):
        async def run(self):
            if self.agent.dataset_id and self.agent.registry:
                X, y, meta = self.agent.registry.load(self.agent.dataset_id)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.agent.test_size)

                # Entrenar el modelo
                self.agent.model.fit(X_train, y_train)

                # Guardar test para explicar después
                self.agent.X_test = X_test
                self.agent.y_test = y_test

                print(f"[{self.agent.jid}] Modelo entrenado automáticamente con dataset '{self.agent.dataset_id}'")

    # ==================== Classification Behaviour ====================
    class ClassificationBehaviour(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg is None:
                return

            # Ignorar si el modelo aún no está entrenado
            if not getattr(self.agent.model, "is_trained", True):
                print(f"[{self.agent.jid}] Modelo aún no entrenado. Ignorando mensaje.")
                return

            try:
                content = json.loads(msg.body)
                action = content.get("action")

                if action != "classify":
                    return

                # Usar datos del mensaje o fallback al test interno
                X = content.get("data", self.agent.X_test)
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
