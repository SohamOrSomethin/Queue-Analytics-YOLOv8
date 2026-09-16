import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd


MODEL_FILE = "xgboost_model.pkl"


class WaitTimePredictor:
    """
    Loads the trained XGBoost model and predicts waiting time.

    Input and output wait-time unit: seconds.
    """

    def __init__(self, model_path=MODEL_FILE):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run 'python train_xgboost.py' first."
            )

        with open(model_path, "rb") as model_file:
            model_payload = pickle.load(model_file)

        self.model = model_payload["model"]
        self.features = model_payload["features"]
        self.wait_time_unit = model_payload.get(
            "wait_time_unit",
            "seconds",
        )

    def predict(
        self,
        hour,
        queue_size,
        recent_avg_wait_time,
    ):
        """
        Predicts customer waiting time in seconds.
        """

        input_data = pd.DataFrame(
            [
                {
                    "hour": int(hour),
                    "queue_size": int(queue_size),
                    "recent_avg_wait_time": float(
                        recent_avg_wait_time
                    ),
                }
            ]
        )

        input_data = input_data[self.features]

        prediction = self.model.predict(
            input_data
        )[0]

        return max(0.0, round(float(prediction), 2))

    def predict_now(
        self,
        queue_size,
        recent_avg_wait_time,
    ):
        """
        Predicts waiting time using the current system hour.
        """

        current_hour = datetime.now().hour

        return self.predict(
            hour=current_hour,
            queue_size=queue_size,
            recent_avg_wait_time=recent_avg_wait_time,
        )


def format_wait_time(wait_seconds):
    """
    Converts seconds into a readable MM:SS string.
    """

    wait_seconds = max(0, int(round(wait_seconds)))

    minutes = wait_seconds // 60
    seconds = wait_seconds % 60

    return f"{minutes} min {seconds:02d} sec"


def main():
    parser = argparse.ArgumentParser(
        description="Predict customer queue waiting time."
    )

    parser.add_argument(
        "--model",
        default=MODEL_FILE,
        help="Path to trained XGBoost model."
    )

    parser.add_argument(
        "--hour",
        type=int,
        default=datetime.now().hour,
        help="Current hour in 24-hour format."
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        required=True,
        help="Current number of people in queue."
    )

    parser.add_argument(
        "--recent-avg-wait",
        type=float,
        required=True,
        help="Recent average waiting time in seconds."
    )

    args = parser.parse_args()

    predictor = WaitTimePredictor(
        args.model
    )

    predicted_wait_seconds = predictor.predict(
        hour=args.hour,
        queue_size=args.queue_size,
        recent_avg_wait_time=args.recent_avg_wait,
    )

    print("\nPrediction Result")
    print("-" * 35)
    print(
        f"Predicted wait: "
        f"{predicted_wait_seconds:.2f} seconds"
    )
    print(
        f"Formatted wait: "
        f"{format_wait_time(predicted_wait_seconds)}"
    )


if __name__ == "__main__":
    main()