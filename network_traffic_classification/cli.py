"""Command-line tools for reproducible traffic classification experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

SUPPORTED_MODELS = ("decision-tree", "random-forest", "knn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="network_traffic_classification",
        description="Train and evaluate network traffic classification models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a classifier from total_class.csv.")
    train.add_argument("--data", required=True, help="Path to the labeled total_class.csv file.")
    train.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="random-forest",
        help="Classifier to train.",
    )
    train.add_argument("--output", help="Optional path for the trained .joblib model.")
    train.add_argument("--labels-output", help="Optional path for the class label list.")
    train.add_argument("--test-size", type=float, default=0.33, help="Test split size.")
    train.add_argument("--random-state", type=int, default=42, help="Random seed.")

    return parser


def train(args: argparse.Namespace) -> None:
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    from .data import encode_labels, load_labeled_flows, split_features_and_labels
    from .models import build_model

    df = load_labeled_flows(args.data)
    x, raw_labels = split_features_and_labels(df)
    y, class_names = encode_labels(raw_labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        shuffle=True,
        stratify=y,
        random_state=args.random_state,
    )

    model = build_model(args.model, random_state=args.random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model: {args.model}")
    print(f"Rows: {len(df)}")
    print(f"Features: {', '.join(x.columns)}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(classification_report(y_test, predictions, target_names=class_names))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "class_names": class_names}, output_path)
        print(f"Saved model: {output_path}")

    if args.labels_output:
        labels_path = Path(args.labels_output)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
        print(f"Saved labels: {labels_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        train(args)
        return

    parser.error(f"Unknown command: {args.command}")
