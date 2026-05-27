"""
ML-based Router using TF-IDF + Naive Bayes Classifier in pure Python.
Determines if a query is a physics or logic problem.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

Route = Literal["logic", "physics"]


class TfidfNaiveBayesClassifier:
    """Pure-Python TF-IDF + Naive Bayes classifier for query classification."""

    def __init__(self):
        self.vocab: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.idf: List[float] = []
        self.class_priors: Dict[str, float] = {}
        self.cond_probs: Dict[str, List[float]] = {}
        self.classes: List[str] = ["physics", "logic"]

    def tokenize(self, text: str) -> List[str]:
        """Convert text into lowercase words of length >= 2."""
        text = text.lower()
        # Keep word characters and numbers
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [w for w in text.split() if len(w) > 1]

    def train(self, documents: List[Tuple[str, str]]) -> None:
        """
        Train Naive Bayes model using TF-IDF weighting on features.
        
        Args:
            documents: List of (text, label) tuples.
        """
        tokenized_docs = []
        df: Dict[str, int] = {}
        word_counts_per_class: Dict[str, Dict[str, int]] = {c: {} for c in self.classes}
        class_docs_count = {c: 0 for c in self.classes}

        for text, label in documents:
            if label not in self.classes:
                continue
            class_docs_count[label] += 1
            words = self.tokenize(text)
            tokenized_docs.append((words, label))

            # Update Document Frequency
            unique_words = set(words)
            for w in unique_words:
                df[w] = df.get(w, 0) + 1

            # Update Word Counts per class
            for w in words:
                word_counts_per_class[label][w] = word_counts_per_class[label].get(w, 0) + 1

        num_docs = len(documents)
        if num_docs == 0:
            return

        # Sort vocab by document frequency, keep top 1200 words, removing standard stop words
        stop_words = {
            "the", "and", "of", "to", "in", "is", "that", "it", "he", "she",
            "they", "a", "an", "on", "for", "with", "as", "by", "at", "are", "be"
        }
        sorted_df = sorted(df.items(), key=lambda x: x[1], reverse=True)
        self.vocab = [w for w, count in sorted_df if w not in stop_words][:1200]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # Compute IDF
        self.idf = []
        for w in self.vocab:
            self.idf.append(math.log(num_docs / (1.0 + df.get(w, 0))))

        # Compute class priors
        for c in self.classes:
            self.class_priors[c] = class_docs_count[c] / num_docs

        # Compute conditional probabilities with Laplace smoothing
        vocab_size = len(self.vocab)
        for c in self.classes:
            total_words_in_class = sum(word_counts_per_class[c].get(w, 0) for w in self.vocab)
            self.cond_probs[c] = []
            for w in self.vocab:
                word_count = word_counts_per_class[c].get(w, 0)
                # Laplace smoothing
                prob = (word_count + 1.0) / (total_words_in_class + vocab_size)
                self.cond_probs[c].append(prob)

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Classify text and return predicted class and confidence.
        """
        if not self.vocab:
            return "logic", 0.5

        words = self.tokenize(text)
        scores = {}
        for c in self.classes:
            score = math.log(self.class_priors.get(c, 0.5))
            for w in words:
                if w in self.word_to_idx:
                    idx = self.word_to_idx[w]
                    # Incorporate TF-IDF by scaling log conditional probabilities with IDF
                    score += math.log(self.cond_probs[c][idx]) * (self.idf[idx] / 10.0)
            scores[c] = score

        # Compute probabilities using softmax-like normalization
        max_score = max(scores.values())
        exp_scores = {c: math.exp(score - max_score) for c, score in scores.items()}
        sum_exp = sum(exp_scores.values())
        probs = {c: exp_scores[c] / sum_exp for c in self.classes}

        best_class = max(probs, key=probs.get)
        return best_class, probs[best_class]

    def save(self, filepath: str) -> None:
        """Save weights to file."""
        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "class_priors": self.class_priors,
            "cond_probs": self.cond_probs,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        """Load weights from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idf = data["idf"]
        self.class_priors = data["class_priors"]
        self.cond_probs = data["cond_probs"]


class RouterAgent:
    """Classifies a unified query stream into Logic or Physics."""

    def __init__(self):
        self.classifier = TfidfNaiveBayesClassifier()
        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        """Load model if cached, otherwise train and cache."""
        root = Path(__file__).resolve().parent.parent
        cache_path = root / "data" / "router_model.json"
        
        if cache_path.exists():
            try:
                self.classifier.load(str(cache_path))
                return
            except Exception:
                pass

        # If cache not found/failed, train on raw datasets
        physics_path = root / "data" / "Physics_Problems_Text_Only_removeQA.json"
        logic_path = root / "data" / "Logic_Based_Educational_Queries.json"

        documents: List[Tuple[str, str]] = []
        
        # Load physics problems
        if physics_path.exists():
            try:
                with open(physics_path, "r", encoding="utf-8") as f:
                    phys_data = json.load(f)
                for item in phys_data:
                    q = item.get("question") or item.get("query")
                    if q:
                        documents.append((str(q), "physics"))
            except Exception:
                pass

        # Load logic problems
        if logic_path.exists():
            try:
                with open(logic_path, "r", encoding="utf-8") as f:
                    logic_data = json.load(f)
                for item in logic_data:
                    qs = item.get("questions")
                    if isinstance(qs, list):
                        for q in qs:
                            if q:
                                documents.append((str(q), "logic"))
                    else:
                        q = item.get("question") or item.get("query")
                        if q:
                            documents.append((str(q), "logic"))
            except Exception:
                pass

        if documents:
            self.classifier.train(documents)
            # Create data folder if it doesn't exist
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.classifier.save(str(cache_path))
            except Exception:
                pass

    def classify(self, payload: Dict[str, Any]) -> Route:
        """
        Determine if the payload refers to logic or physics.
        Prioritizes explicit payload types, falls back to TF-IDF classifier.
        """
        # Prioritize explicit labels
        t = str(payload.get("type") or payload.get("query_type") or "").lower()
        if t in {"logic", "type1", "type_1", "educational_logic", "edu"}:
            return "logic"
        if t in {"physics", "type2", "type_2"}:
            return "physics"

        # Explicit metadata indicators
        if payload.get("premises-NL") or payload.get("premises") or payload.get("premises_nl"):
            return "logic"
        if payload.get("premises-FOL") or payload.get("premises_fol"):
            return "logic"
        if "unit" in payload:
            return "physics"

        record_id = str(payload.get("id", "")).upper()
        if record_id.startswith(("TD", "LD", "CH", "DD", "DT", "NL", "TH")):
            return "physics"

        # Content classification
        q = str(payload.get("question") or payload.get("query") or "")
        best_class, _ = self.classifier.predict(q)
        return best_class

    def classify_with_confidence(self, q: str) -> Tuple[Route, float]:
        """Classify and return predicted route + confidence probability."""
        best_class, conf = self.classifier.predict(q)
        return best_class, conf
