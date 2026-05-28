from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_TRAIN_PATH = Path("data/split/train.json")
DEFAULT_OUT_DIR = Path("models/tf-idf")


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


def extract_questions(records: list[dict[str, Any]], source: Path) -> list[str]:
    return [normalize_question(record, index, source) for index, record in enumerate(records)]


def build_vectorizer(
    min_df: int,
    max_df: float,
    ngram_range: tuple[int, int],
) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
    )


def save_vocab(vectorizer: TfidfVectorizer, out_dir: Path, train_path: Path, train_count: int) -> None:
    vocab = dict(sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1]))
    payload = {
        "train_path": str(train_path),
        "train_count": train_count,
        "vocabulary_size": len(vocab),
        "vocabulary": vocab,
        "vectorizer": {
            "lowercase": vectorizer.lowercase,
            "strip_accents": vectorizer.strip_accents,
            "analyzer": vectorizer.analyzer,
            "token_pattern": vectorizer.token_pattern,
            "ngram_range": list(vectorizer.ngram_range),
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "sublinear_tf": vectorizer.sublinear_tf,
            "norm": vectorizer.norm,
        },
    }

    with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TF-IDF vocabulary from split train data.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH, help="Path to train.json.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for TF-IDF artifacts.")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=1.0, help="Maximum document frequency.")
    parser.add_argument("--ngram-min", type=int, default=1, help="Minimum n-gram length.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Maximum n-gram length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        raise ValueError("--ngram-min must be >= 1 and --ngram-max must be >= --ngram-min")

    records = load_json(args.train)
    questions = extract_questions(records, args.train)

    vectorizer = build_vectorizer(
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    vectorizer.fit(questions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, args.out_dir / "vectorizer.joblib")
    save_vocab(vectorizer, args.out_dir, args.train, len(questions))

    print(
        json.dumps(
            {
                "train_count": len(questions),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "vectorizer": str(args.out_dir / "vectorizer.joblib"),
                "vocab": str(args.out_dir / "vocab.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
