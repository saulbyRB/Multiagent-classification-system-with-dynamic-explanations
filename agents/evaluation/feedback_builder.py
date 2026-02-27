class FeedbackBuilder:
    def build(self, agent_id, decision, evaluation, explanations=None, idx=0):
        # "action" no debería ir aquí, ya lo añade el aggregator
        return {
            "strategy": decision,
            "agent_id": agent_id,   # útil para logging/debug
            "evaluation": {
                "reward": evaluation["scores"][idx],
                "prediction_alignment": evaluation["components"]["pred"][idx],
                "confidence": evaluation["components"]["conf"][idx],
                "explanation_similarity": evaluation["components"]["exp"][idx],
                "exp_quality": evaluation["components"]["exp_detail"]["quality"][idx],
                "exp_consensus": evaluation["components"]["exp_detail"]["consensus"][idx],
                "exp_stability": evaluation["components"]["exp_detail"]["stability"][idx],
                "num_explainers": len(explanations) if explanations else 0,
                "global_score": evaluation["global_score"],
            }
        }