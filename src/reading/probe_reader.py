"""Linear probe reader for CI-Steering, with cross-task evaluation for privacy awareness gap."""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import Optional


class ProbeReader:
    """
    Supervised linear probe reader for privacy representations.

    For each layer, trains a logistic regression classifier to predict
    whether the activation corresponds to a contextually appropriate
    or inappropriate information flow.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        cv_folds: int = 5,
    ):
        self.max_iter = max_iter
        self.C = C
        self.cv_folds = cv_folds

        self.probes = {}           # layer -> fitted LogisticRegression
        self.scalers = {}          # layer -> fitted StandardScaler
        self.layer_scores = {}     # layer -> {accuracy, auroc, cv_mean, cv_std}
        self.coef_directions = {}  # layer -> weight vector (privacy direction)

    def fit(
        self,
        activations: dict[int, torch.Tensor],
        labels: list[bool],
    ):
        """Train a linear probe at each layer."""
        labels_arr = np.array(labels, dtype=int)

        for layer_idx, acts_tensor in activations.items():
            acts = acts_tensor.float().numpy()

            scaler = StandardScaler()
            acts_scaled = scaler.fit_transform(acts)
            self.scalers[layer_idx] = scaler

            probe = LogisticRegression(
                max_iter=self.max_iter,
                C=self.C,
                solver="lbfgs",
                random_state=42,
            )
            probe.fit(acts_scaled, labels_arr)
            self.probes[layer_idx] = probe

            # coef_ lives in scaled space; divide by scale_ to get direction in raw activation space
            self.coef_directions[layer_idx] = (probe.coef_[0] / scaler.scale_).copy()

            # Pipeline ensures each CV fold fits its own scaler
            cv_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=self.max_iter, C=self.C,
                    solver="lbfgs", random_state=42,
                )),
            ])
            cv_scores = cross_val_score(
                cv_pipeline,
                acts, labels_arr,  # raw activations — Pipeline handles scaling
                cv=min(self.cv_folds, len(labels_arr)),
                scoring="accuracy",
            )

            train_pred = probe.predict(acts_scaled)
            train_proba = probe.predict_proba(acts_scaled)[:, 1]
            train_acc = accuracy_score(labels_arr, train_pred)

            try:
                train_auroc = roc_auc_score(labels_arr, train_proba)
            except ValueError:
                train_auroc = 0.5

            self.layer_scores[layer_idx] = {
                "train_accuracy": float(train_acc),
                "train_auroc": float(train_auroc),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
            }

    def evaluate(
        self,
        activations: dict[int, torch.Tensor],
        labels: list[bool],
    ) -> dict[int, dict]:
        """
        Evaluate probes on held-out data.

        Returns:
            dict mapping layer_idx -> {accuracy, auroc, predictions, probabilities}
        """
        labels_arr = np.array(labels, dtype=int)
        results = {}

        for layer_idx, acts_tensor in activations.items():
            if layer_idx not in self.probes:
                continue

            acts = acts_tensor.float().numpy()
            acts_scaled = self.scalers[layer_idx].transform(acts)
            probe = self.probes[layer_idx]

            predictions = probe.predict(acts_scaled)
            probabilities = probe.predict_proba(acts_scaled)[:, 1]

            acc = accuracy_score(labels_arr, predictions)
            try:
                auroc = roc_auc_score(labels_arr, probabilities)
            except ValueError:
                auroc = 0.5

            results[layer_idx] = {
                "accuracy": float(acc),
                "auroc": float(auroc),
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
            }

        return results

    def get_best_layers(self, top_k: int = 5, metric: str = "cv_mean") -> list[int]:
        """Return the top-k layers with highest probe accuracy."""
        sorted_layers = sorted(
            self.layer_scores.items(),
            key=lambda x: x[1][metric],
            reverse=True,
        )
        return [layer for layer, score in sorted_layers[:top_k]]

    def get_privacy_direction(self, layer_idx: int) -> np.ndarray:
        """
        Get the learned privacy direction (probe weight vector) for a layer.
        This is the supervised alternative to PCA-based direction.
        """
        if layer_idx not in self.coef_directions:
            raise KeyError(f"No probe found for layer {layer_idx}")
        direction = self.coef_directions[layer_idx]
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm
        return direction

    def evaluate_cross_task(
        self,
        activations: dict[int, torch.Tensor],
        labels: list[bool],
    ) -> dict[int, dict]:
        """Apply concept-trained probe to a different task's activations (no retraining)."""
        return self.evaluate(activations, labels)

    def compute_privacy_awareness_gap(
        self,
        cross_task_results: dict[int, dict],
        behavioral_ncr: float,
        best_layer: Optional[int] = None,
    ) -> dict:
        """Compute gap between cross-task probe accuracy and behavioral NCR."""
        if best_layer is None:
            best_layer = self.get_best_layers(top_k=1)[0]

        probe_acc = cross_task_results[best_layer]["accuracy"]
        probe_auroc = cross_task_results[best_layer]["auroc"]

        gap = probe_acc - behavioral_ncr

        return {
            "best_layer": best_layer,
            "cross_task_probe_accuracy": probe_acc,
            "cross_task_probe_auroc": probe_auroc,
            "behavioral_ncr": behavioral_ncr,
            "privacy_awareness_gap": gap,
            "interpretation": (
                f"A probe trained on concept-level vignettes classifies the "
                f"behavioral-task activations with {probe_acc:.1%} accuracy "
                f"(layer {best_layer}), confirming that the model encodes the "
                f"privacy norm during the exact task where it must act. "
                f"Yet it complies only {behavioral_ncr:.1%} of the time. "
                f"This {gap:.1%} gap shows the model 'knows' the norm at "
                f"generation time but fails to act on it."
            ),
        }

    def save(self, path: str):
        """Save fitted probes to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save probe directions
        directions = {
            str(k): torch.from_numpy(v) for k, v in self.coef_directions.items()
        }
        torch.save(directions, save_dir / "probe_directions.pt")

        # Save scores
        with open(save_dir / "probe_scores.json", "w") as f:
            json.dump({str(k): v for k, v in self.layer_scores.items()}, f, indent=2)

        print(f"Saved probe reader to {save_dir}")

    def load(self, path: str):
        """Load probes from disk (directions and scores only; probes need retraining)."""
        load_dir = Path(path)

        directions = torch.load(
            load_dir / "probe_directions.pt", map_location="cpu", weights_only=True
        )
        self.coef_directions = {int(k): v.numpy() for k, v in directions.items()}

        with open(load_dir / "probe_scores.json", "r") as f:
            scores = json.load(f)
        self.layer_scores = {int(k): v for k, v in scores.items()}

        print(f"Loaded probe reader from {load_dir}")
