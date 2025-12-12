import joblib
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightMLFilter:
    
    def __init__(self, model_path: str = "training/plan_detection_model.pkl", threshold: float = 0.55):
       
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"ML Filter model loaded from {self.model_path}")
            logger.info(f"Model classes: {self.model.classes_}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            logger.error("Please run 'python train_light_filter.py' first to train the model")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        if not self.model:
            raise RuntimeError("Model not loaded")

        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        plan_idx = list(classes).index("plan") if "plan" in classes else 0
        plan_prob = float(probabilities[plan_idx])

        is_plan = plan_prob >= self.threshold
        prediction = "plan" if is_plan else "noise"

        return {
            "text": text,
            "confidence": plan_prob,
            "is_plan": is_plan,
            "prediction": prediction,
            "threshold": self.threshold
        }

    def filter_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        text = message.get("text_clean", message.get("text", ""))

        if not text or len(text.strip()) == 0:
            message.update({
                "confidence": 0.0,
                "is_plan": False,
                "prediction": "noise",
                "ml_reason": "Empty text",
                "score": 0.0,
                "tag": "noise"
            })
            return message

        result = self.predict_single(text)

        message.update({
            "confidence": result["confidence"],
            "is_plan": result["is_plan"],
            "prediction": result["prediction"],
            "ml_reason": f"ML confidence: {result['confidence']:.3f}",
            "score": result["confidence"],
            "tag": "potential_plan" if result["is_plan"] else "noise"
        })

        return message

    def __call__(self, message: Dict[str, Any]) -> Dict[str, Any]:
        return self.filter_message(message)


def create_ml_filter_node(model_path: str = "training/plan_detection_model.pkl", threshold: float = 0.55):
    ml_filter = LightweightMLFilter(model_path, threshold)

    def ml_filter_node(state):
        processed_messages = state.get("processed_messages", [])
        potential_plans = []
        discarded_messages = []

        logger.info(f"ML Filter processing {len(processed_messages)} messages")

        for message in processed_messages:
            if not message.get('is_valid', True):
                message.update({
                    "confidence": 0.0,
                    "is_plan": False,
                    "tag": "invalid",
                    "ml_reason": "Invalid message"
                })
                discarded_messages.append(message)
                continue

            filtered_message = ml_filter.filter_message(message)

            if filtered_message["is_plan"]:
                potential_plans.append(filtered_message)
            else:
                discarded_messages.append(filtered_message)

        state["potential_plans"] = potential_plans
        state["discarded_messages"] = discarded_messages
        state["current_node"] = "ml_filter"
        state["pipeline_status"] = "ml_filter_complete"

        logger.info(f"ML Filter completed: {len(potential_plans)} potential plans, {len(discarded_messages)} discarded")

        return state

    return ml_filter_node