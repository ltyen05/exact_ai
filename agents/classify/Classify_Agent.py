"""TF-IDF plus logistic-regression router for logic and physics questions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TfidfLogisticClassifier:
    """Classify a question with the trained TF-IDF routing artifacts."""

    DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "tf-idf"
    LABEL_TO_ROUTE = {"1": "logic", "2": "physics"}

    def __init__(
        self,
        model_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Configure the directory that contains the saved routing model."""
        self.config = config or {}
        configured_model_dir = self.config.get("model_dir")
        self.model_dir = Path(model_dir or configured_model_dir or self.DEFAULT_MODEL_DIR)
        self.vectorizer: Any | None = None
        self.model: Any | None = None

    def initialize(self) -> None:
        """Load the trained vectorizer and classifier from disk once."""
        try:
            from joblib import load
        except ImportError as exc:
            raise RuntimeError("Routing requires joblib and scikit-learn to be installed.") from exc

        vectorizer_path = self.model_dir / "vectorizer.joblib"
        model_path = self.model_dir / "model.joblib"
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Missing vectorizer: {vectorizer_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        self.vectorizer = load(vectorizer_path)
        self.model = load(model_path)

    def validate_output(self, output_data: Any) -> bool:
        """Check that a classifier result contains a supported workflow route."""
        return (
            isinstance(output_data, dict)
            and isinstance(output_data.get("question"), str)
            and bool(output_data["question"].strip())
            and output_data.get("Type") in {"logic", "physics"}
        )

    def run(self, input_data: Any) -> dict[str, Any]:
        """Predict only the workflow route required for orchestration."""
        question = str(input_data).strip()
        if not question:
            raise ValueError("question must be non-empty")
        if self.vectorizer is None or self.model is None:
            self.initialize()

        matrix = self.vectorizer.transform([question])
        raw_type = str(self.model.predict(matrix)[0])
        route = self.LABEL_TO_ROUTE.get(raw_type)
        if route is None:
            raise ValueError(f"Unsupported classifier label: {raw_type}")
        return {
            "question": question,
            "Type": route,
        }
