from __future__ import annotations

import json
from pathlib import Path

from joblib import load


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "models" / "tf-idf"


def load_model(model_dir: Path = MODEL_DIR):
    vectorizer_path = model_dir / "vectorizer.joblib"
    model_path = model_dir / "model.joblib"
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Missing vectorizer: {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    vectorizer = load(vectorizer_path)
    model = load(model_path)
    return vectorizer, model


def classify_question(question: str, model_dir: Path = MODEL_DIR) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")

    vectorizer, model = load_model(model_dir)
    x_question = vectorizer.transform([question])

    predicted = str(model.predict(x_question)[0])
    probabilities = model.predict_proba(x_question)[0]
    class_to_probability = {
        str(label): float(probabilities[index])
        for index, label in enumerate(model.classes_)
    }

    return {
        "question": question,
        "Type": predicted,
        "confidence": class_to_probability[predicted],
        "probabilities": class_to_probability,
    }


def test_physics_question():
    question = "What is the total resistance of two 10\\Omega and 30\\Omega branches connected in parallel?"
    result = classify_question(question)
    assert result["Type"] == "2", result


def test_logic_question():
    question = "The statement that if a student learns, they develop critical thinking, problem-solving skills, and growth follows because learning leads to critical thinking and problem-solving (premise 10) and, since everyone is a student (premise 3), learning also leads to growth (premise 1)."
    result = classify_question(question)
    assert result["Type"] == "1", result


if __name__ == "__main__":
    results = [
        classify_question("What is the total resistance of two 10\\Omega and 30\\Omega branches connected in parallel?"),
        classify_question(
            "The statement that if a student learns, they develop critical thinking, problem-solving skills, "
            "and growth follows because learning leads to critical thinking and problem-solving (premise 10) "
            "and, since everyone is a student (premise 3), learning also leads to growth (premise 1)."
        ),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
