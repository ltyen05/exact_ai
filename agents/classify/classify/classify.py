from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


DEFAULT_DATA_DIR = Path("data/split")
DEFAULT_MODEL_DIR = Path("models/tf-idf")
LABELS = ["1", "2"]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    return data


def normalize_question(record: dict[str, Any], index: int, source: Path) -> str:
    question = record.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()

    questions = record.get("questions")
    if isinstance(questions, list):
        clean_questions = [str(item).strip() for item in questions if str(item).strip()]
        if clean_questions:
            return "\n".join(clean_questions)

    raise ValueError(f"Missing non-empty question/questions at {source}:{index}")


def normalize_label(record: dict[str, Any], index: int, source: Path) -> str:
    label = str(record.get("Type", "")).strip()
    if label not in LABELS:
        raise ValueError(f"Invalid Type at {source}:{index}: expected one of {LABELS}, got {label!r}")
    return label


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    records = load_json(path)
    questions = [normalize_question(record, index, path) for index, record in enumerate(records)]
    labels = [normalize_label(record, index, path) for index, record in enumerate(records)]
    return questions, labels


def evaluate_split(model: LogisticRegression, x_matrix: Any, y_true: list[str]) -> dict[str, Any]:
    y_pred = model.predict(x_matrix)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    return {
        "count": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro": precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[:3],
        "weighted": precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[:3],
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        ),
    }


def make_json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def train(args: argparse.Namespace) -> None:
    vectorizer_path = args.model_dir / "vectorizer.joblib"
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Missing {vectorizer_path}. Run build_vocab.py before training the classifier."
        )

    vectorizer = load(vectorizer_path)
    train_questions, train_labels = load_dataset(args.data_dir / "train.json")
    valid_questions, valid_labels = load_dataset(args.data_dir / "valid.json")
    test_questions, test_labels = load_dataset(args.data_dir / "test.json")

    x_train = vectorizer.transform(train_questions)
    x_valid = vectorizer.transform(valid_questions)
    x_test = vectorizer.transform(test_questions)

    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    
    model.fit(x_train, train_labels)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, args.model_dir / "model.joblib")

    label_map = {
        "labels": LABELS,
        "model_classes": [str(label) for label in model.classes_],
        "output_field": "Type",
    }
    metrics = {
        "train": evaluate_split(model, x_train, train_labels),
        "valid": evaluate_split(model, x_valid, valid_labels),
        "test": evaluate_split(model, x_test, test_labels),
        "artifacts": {
            "vectorizer": str(vectorizer_path),
            "model": str(args.model_dir / "model.joblib"),
            "label_map": str(args.model_dir / "label_map.json"),
            "metrics": str(args.model_dir / "metrics.json"),
        },
        "config": {
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
    }

    with (args.model_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with (args.model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(metrics), f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "model": str(args.model_dir / "model.joblib"),
                "valid_accuracy": metrics["valid"]["accuracy"],
                "valid_macro": metrics["valid"]["macro"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_macro": metrics["test"]["macro"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def predict(args: argparse.Namespace) -> None:
    vectorizer = load(args.model_dir / "vectorizer.joblib")
    model = load(args.model_dir / "model.joblib")

    question = args.question.strip()
    if not question:
        raise ValueError("--question must be non-empty")

    x_question = vectorizer.transform([question])
    predicted = str(model.predict(x_question)[0])
    probabilities = model.predict_proba(x_question)[0]
    class_to_probability = {
        str(label): float(probabilities[index])
        for index, label in enumerate(model.classes_)
    }
    confidence = class_to_probability[predicted]

    print(
        json.dumps(
            {
                "question": question,
                "Type": predicted,
                "confidence": confidence,
                "probabilities": class_to_probability,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate/predict Type with TF-IDF + sigmoid.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the classifier.")
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with split JSON files.")
    train_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory with model artifacts.")
    train_parser.add_argument("--max-iter", type=int, default=1000, help="LogisticRegression max_iter.")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    predict_parser = subparsers.add_parser("predict", help="Predict Type for one question.")
    predict_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory with model artifacts.")
    predict_parser.add_argument("--question", required=True, help="Input question text.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
