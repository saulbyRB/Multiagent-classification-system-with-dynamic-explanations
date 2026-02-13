# feedback_builder.py

class FeedbackBuilder:
    """
    Construye mensajes de feedback RL para cada clasificador
    """

    def build(self, agent_id, decision, evaluation, explanations=None, idx=0):
        return {
            "action": "feedback",
            "strategy": decision,
            "evaluation": {
                "reward": evaluation["scores"][idx],
                "prediction_alignment": evaluation["components"]["pred"][idx],
                "confidence": evaluation["components"]["conf"][idx],
                "explanation_similarity": evaluation["components"]["exp"][idx],
                "num_explainers": len(explanations) if explanations else 0,
                "global_score": evaluation["global_score"]
            }
        }
