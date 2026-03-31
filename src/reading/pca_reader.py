"""PCA-based representation reading for CI-Steering (LAT methodology, Zou et al. 2023)."""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Optional


class PCAReader:
    """
    Unsupervised representation reader using PCA on paired differences
    to find privacy directions, matching the original RepE methodology.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        # Per-layer results
        self.privacy_directions = {}       # layer -> (hidden_dim,) unit vector
        self.direction_signs = {}          # layer -> +1 or -1
        self.mean_differences = {}         # layer -> (hidden_dim,) raw mean diff
        self.H_train_means = {}            # layer -> mean of difference vectors (for recentering)
        self.pca_models = {}               # layer -> fitted PCA object
        self.layer_scores = {}             # layer -> classification accuracy

    def fit(
        self,
        activations: dict[int, torch.Tensor],
        labels: list[bool],
        pair_ids: Optional[list[int]] = None,
        method: str = "pca",
    ):
        """Fit PCA on paired differences h(appropriate) - h(inappropriate).

        Sign convention: positive projection = "appropriate".
        """
        labels_arr = np.array(labels, dtype=bool)

        # ---- Build paired differences (core LAT methodology) ----
        if pair_ids is not None:
            paired_diffs, pair_labels = self._compute_paired_differences(
                activations, labels_arr, pair_ids
            )
            use_paired = True
        else:
            paired_diffs = None
            use_paired = False

        for layer_idx, acts_tensor in activations.items():
            acts = acts_tensor.float().numpy()

            mean_app = acts[labels_arr].mean(axis=0)
            mean_inapp = acts[~labels_arr].mean(axis=0)
            mean_diff = mean_app - mean_inapp
            self.mean_differences[layer_idx] = mean_diff

            if method in ("pca", "both"):
                if use_paired and paired_diffs is not None:
                    diffs = paired_diffs[layer_idx]  # shape (n_pairs, hidden_dim)
                    diff_mean = diffs.mean(axis=0, keepdims=True)
                    self.H_train_means[layer_idx] = diff_mean
                    diffs_centered = diffs - diff_mean

                    n_comp = min(self.n_components, diffs_centered.shape[0], diffs_centered.shape[1])
                    pca = PCA(n_components=n_comp, whiten=False)
                    pca.fit(diffs_centered)
                    self.pca_models[layer_idx] = pca

                    # First PC of the differences = privacy direction
                    pc1 = pca.components_[0].astype(np.float32)

                    # Determine sign: project paired diffs onto pc1
                    # Positive should mean "appropriate > inappropriate"
                    projections = diffs @ pc1
                    # If most diffs project positively, sign is correct
                    # (since diffs = appropriate - inappropriate)
                    sign = 1 if np.mean(projections) >= 0 else -1
                    pc1 = sign * pc1

                    self.privacy_directions[layer_idx] = pc1
                    self.direction_signs[layer_idx] = sign

                else:
                    # Fallback: PCA on raw activations (less correct)
                    pca = PCA(n_components=min(self.n_components, acts.shape[0], acts.shape[1]))
                    pca.fit(acts)
                    self.pca_models[layer_idx] = pca

                    pc1 = pca.components_[0].astype(np.float32)

                    projections = acts @ pc1
                    proj_app = projections[labels_arr].mean()
                    proj_inapp = projections[~labels_arr].mean()
                    if proj_app < proj_inapp:
                        pc1 = -pc1

                    self.privacy_directions[layer_idx] = pc1

            elif method == "mean_diff":
                norm = np.linalg.norm(mean_diff)
                if norm > 0:
                    self.privacy_directions[layer_idx] = (mean_diff / norm).astype(np.float32)
                else:
                    self.privacy_directions[layer_idx] = mean_diff.astype(np.float32)

            # Compute accuracy using the privacy direction on raw activations
            direction = self.privacy_directions[layer_idx]
            projections = acts @ direction
            threshold = np.median(projections)
            predictions = projections > threshold
            acc = accuracy_score(labels_arr, predictions)

            try:
                auroc = roc_auc_score(labels_arr.astype(int), projections)
            except ValueError:
                auroc = 0.5

            self.layer_scores[layer_idx] = {
                "accuracy": float(acc),
                "auroc": float(auroc),
            }

    def _compute_paired_differences(
        self,
        activations: dict[int, torch.Tensor],
        labels: np.ndarray,
        pair_ids: list[int],
    ) -> tuple[dict, list]:
        """Compute h(appropriate) - h(inappropriate) for each pair_id."""
        # Build pair mapping: pair_id -> {True: index, False: index}
        pair_map = {}
        for idx, (pid, label) in enumerate(zip(pair_ids, labels)):
            if pid not in pair_map:
                pair_map[pid] = {}
            pair_map[pid][bool(label)] = idx

        # Only keep complete pairs
        valid_pairs = [
            pid for pid, mapping in pair_map.items()
            if True in mapping and False in mapping
        ]
        valid_pairs.sort()  # deterministic ordering

        paired_diffs = {}
        for layer_idx, acts_tensor in activations.items():
            acts = acts_tensor.float().numpy()
            diffs = []
            for pid in valid_pairs:
                app_idx = pair_map[pid][True]
                inapp_idx = pair_map[pid][False]
                diffs.append(acts[app_idx] - acts[inapp_idx])
            paired_diffs[layer_idx] = np.stack(diffs, axis=0)

        return paired_diffs, valid_pairs

    def predict(
        self,
        activations: dict[int, torch.Tensor],
        layer_idx: Optional[int] = None,
    ) -> dict[int, np.ndarray]:
        """Project activations onto privacy directions.

        Positive = predicted appropriate, negative = predicted inappropriate.
        """
        results = {}
        target_layers = [layer_idx] if layer_idx is not None else list(self.privacy_directions.keys())

        for l_idx in target_layers:
            if l_idx not in self.privacy_directions:
                continue
            direction = self.privacy_directions[l_idx]
            acts = activations[l_idx].float().numpy()
            projections = acts @ direction
            results[l_idx] = projections

        return results

    def evaluate(
        self,
        activations: dict[int, torch.Tensor],
        labels: list[bool],
    ) -> dict[int, dict]:
        """Evaluate reading accuracy on held-out data."""
        labels_arr = np.array(labels, dtype=bool)
        projections = self.predict(activations)

        results = {}
        for layer_idx, proj in projections.items():
            threshold = np.median(proj)
            predictions = proj > threshold
            acc = accuracy_score(labels_arr, predictions)

            try:
                auroc = roc_auc_score(labels_arr.astype(int), proj)
            except ValueError:
                auroc = 0.5

            results[layer_idx] = {
                "accuracy": float(acc),
                "auroc": float(auroc),
                "threshold": float(threshold),
            }

        return results

    def evaluate_multi_component(
        self,
        activations: dict[int, torch.Tensor],
        labels: list[bool],
        k_values: list[int] = [1, 3, 5, 10],
    ) -> dict[int, dict]:
        """Evaluate classification using top-k PCs + logistic regression."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score

        labels_arr = np.array(labels, dtype=bool)
        results = {}

        for layer_idx in activations:
            if layer_idx not in self.pca_models:
                continue

            acts = activations[layer_idx].float().numpy()
            pca = self.pca_models[layer_idx]

            layer_results = {}
            for k in k_values:
                k_actual = min(k, pca.n_components_)
                # Project onto top-k PCs
                X_proj = acts @ pca.components_[:k_actual].T  # (N, k)

                # Train logistic regression
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_proj)

                clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
                clf.fit(X_scaled, labels_arr.astype(int))

                preds = clf.predict(X_scaled)
                probs = clf.predict_proba(X_scaled)[:, 1]

                acc = accuracy_score(labels_arr.astype(int), preds)
                try:
                    auroc = roc_auc_score(labels_arr.astype(int), probs)
                except ValueError:
                    auroc = 0.5

                layer_results[k] = {"accuracy": float(acc), "auroc": float(auroc)}

            results[layer_idx] = layer_results

        return results

    def get_best_layers(self, top_k: int = 5, metric: str = "auroc") -> list[int]:
        """Return the top-k layers with highest reading accuracy."""
        sorted_layers = sorted(
            self.layer_scores.items(),
            key=lambda x: x[1][metric] if isinstance(x[1], dict) else x[1],
            reverse=True,
        )
        return [layer for layer, score in sorted_layers[:top_k]]

    def get_privacy_vector(self, layer_idx: int) -> np.ndarray:
        """Get the privacy direction vector for a specific layer."""
        if layer_idx not in self.privacy_directions:
            raise KeyError(f"No privacy direction found for layer {layer_idx}")
        return self.privacy_directions[layer_idx]

    def get_variance_explained(self) -> dict[int, list[float]]:
        """Return PCA variance explained ratios for each layer."""
        return {
            layer: pca.explained_variance_ratio_.tolist()
            for layer, pca in self.pca_models.items()
        }

    def get_pc_ci_alignment(
        self,
        ci_directions: dict[str, np.ndarray],
        layer_idx: int,
    ) -> dict[str, list[float]]:
        """Cosine similarity of CI-parameter directions with top PCs at a given layer."""
        if layer_idx not in self.pca_models:
            return {}

        pca = self.pca_models[layer_idx]
        results = {}

        for name, direction in ci_directions.items():
            d = direction / (np.linalg.norm(direction) + 1e-10)
            alignments = []
            for i in range(min(10, pca.n_components_)):
                pc = pca.components_[i]
                cos_sim = float(np.dot(d, pc))
                alignments.append(cos_sim)
            results[name] = alignments

        return results

    def save(self, path: str):
        """Save fitted reader to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save privacy directions as a tensor dict
        directions = {
            str(k): torch.from_numpy(v) for k, v in self.privacy_directions.items()
        }
        torch.save(directions, save_dir / "privacy_directions.pt")

        # Save scores
        with open(save_dir / "layer_scores.json", "w") as f:
            json.dump({str(k): v for k, v in self.layer_scores.items()}, f, indent=2)

        # Save variance explained
        var_explained = self.get_variance_explained()
        with open(save_dir / "variance_explained.json", "w") as f:
            json.dump({str(k): v for k, v in var_explained.items()}, f, indent=2)

        print(f"Saved PCA reader to {save_dir}")

    def load(self, path: str):
        """Load a fitted reader from disk."""
        load_dir = Path(path)

        # Load privacy directions
        directions = torch.load(
            load_dir / "privacy_directions.pt", map_location="cpu", weights_only=True
        )
        self.privacy_directions = {int(k): v.numpy() for k, v in directions.items()}

        # Load scores
        with open(load_dir / "layer_scores.json", "r") as f:
            scores = json.load(f)
        self.layer_scores = {int(k): v for k, v in scores.items()}

        print(f"Loaded PCA reader from {load_dir}")
