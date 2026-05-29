"""Model factory for supported classical ML experiments."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_MODELS = ("decision-tree", "random-forest", "knn")


def build_model(model_name: str, random_state: int):
    """Create a classifier using the original experiment defaults where possible."""
    if model_name == "decision-tree":
        return DecisionTreeClassifier(random_state=random_state)

    if model_name == "random-forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=random_state,
        )

    if model_name == "knn":
        return KNeighborsClassifier(n_neighbors=3)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}.")
