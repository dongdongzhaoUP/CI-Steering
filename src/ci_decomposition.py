#!/usr/bin/env python3

import argparse
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.activation_extractor import ActivationExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_pca(X: torch.Tensor, n_components: int = 5):
    """PCA via SVD on GPU. Returns components and variance ratios."""
    X_centered = X - X.mean(dim=0, keepdim=True)
    n_components = min(n_components, X.shape[0], X.shape[1])
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    explained_var = (S[:n_components] ** 2) / (X.shape[0] - 1)
    total_var = explained_var.sum() + ((S[n_components:] ** 2) / (X.shape[0] - 1)).sum()
    variance_ratio = (explained_var / total_var).cpu().tolist()
    return components, variance_ratio


def gpu_linear_probe(X: torch.Tensor, labels: torch.Tensor, n_classes: int,
                     epochs: int = 200, lr: float = 0.01):
    """Fast linear probe on GPU using cross-entropy."""
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_norm = (X - mean) / std

    probe = nn.Linear(X_norm.shape[1], n_classes).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        logits = probe(X_norm)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_norm).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return acc


def compute_direction_for_parameter(
    activations: dict[int, torch.Tensor],
    varied_values: list[str],
) -> dict[int, dict]:
    """Compute PCA direction and linear probe for a single CI parameter."""
    encoder = LabelEncoder()
    labels_np = encoder.fit_transform(varied_values)
    labels_t = torch.tensor(labels_np, dtype=torch.long, device=DEVICE)
    n_classes = len(encoder.classes_)

    layer_results = {}

    for layer_idx, acts_tensor in activations.items():
        acts_gpu = acts_tensor.float().to(DEVICE)

        components, variance_ratio = gpu_pca(acts_gpu, n_components=5)
        direction = components[0]  # still on GPU

        # Canonical sign: SVD signs are arbitrary; fix so mean projection > 0.
        projections = acts_gpu @ direction
        if projections.mean() < 0:
            direction = -direction

        direction = direction.cpu().numpy()

        acc = gpu_linear_probe(acts_gpu, labels_t, n_classes)

        layer_results[layer_idx] = {
            "direction": direction,
            "probe_accuracy": float(acc),
            "pca_variance_explained": variance_ratio,
            "n_classes": n_classes,
            "class_names": encoder.classes_.tolist(),
        }

    return layer_results


def compute_cosine_similarity_matrix(
    param_directions: dict[str, dict[int, dict]],
    layer_idx: int,
) -> dict:
    """Cosine similarities between CI parameter directions at a given layer."""
    param_names = list(param_directions.keys())
    n = len(param_names)
    sim_matrix = np.zeros((n, n))

    directions = {}
    for name in param_names:
        if layer_idx in param_directions[name]:
            d = param_directions[name][layer_idx]["direction"]
            d = d / (np.linalg.norm(d) + 1e-10)
            directions[name] = d

    for i, name_i in enumerate(param_names):
        for j, name_j in enumerate(param_names):
            if name_i in directions and name_j in directions:
                sim = np.dot(directions[name_i], directions[name_j])
                sim_matrix[i, j] = sim

    return {
        "parameters": param_names,
        "similarity_matrix": sim_matrix.tolist(),
        "layer": layer_idx,
    }


def permutation_test_cosine_similarity(
    all_activations: dict[str, dict[int, torch.Tensor]],
    all_varied_values: dict[str, list[str]],
    real_cosine_sims: dict,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for CI parameter direction independence.

    Pools stimuli, randomly partitions into groups, computes PCA directions,
    and compares pairwise |cos| against the real CI decomposition.
    """
    import random as rng_module
    rng = rng_module.Random(seed)
    
    # Pool activations across parameters
    param_names = list(all_activations.keys())
    param_sizes = {p: next(iter(all_activations[p].values())).shape[0] for p in param_names}
    layers = sorted(next(iter(all_activations.values())).keys())
    
    # Pick analysis layer: use same as real cosine similarity analysis
    # Use middle-upper layer as representative
    analysis_layer = layers[len(layers) * 3 // 4]
    
    # Pool activations at analysis layer
    pooled_acts = []
    for p in param_names:
        acts = all_activations[p][analysis_layer].float()
        pooled_acts.append(acts)
    pooled = torch.cat(pooled_acts, dim=0)  # (total_N, hidden_dim)
    total_n = pooled.shape[0]
    
    # Group sizes (match original parameter group sizes)
    sizes = [param_sizes[p] for p in param_names]
    
    null_distribution = []
    
    for perm_i in range(n_permutations):
        # Random assignment to groups
        indices = list(range(total_n))
        rng.shuffle(indices)
        
        groups = []
        start = 0
        for size in sizes:
            group_indices = indices[start:start + size]
            groups.append(group_indices)
            start += size
        
        # Compute PCA direction for each group
        directions = []
        for group_idx in groups:
            group_acts = pooled[group_idx].to(DEVICE)
            components, _ = gpu_pca(group_acts, n_components=1)
            direction = components[0].cpu().numpy()
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            directions.append(direction)
        
        # Compute pairwise cosine similarities
        cos_sims = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                cos = abs(float(np.dot(directions[i], directions[j])))
                cos_sims.append(cos)
        
        null_distribution.append(np.mean(cos_sims))
    
    # Compute real mean |cos|
    real_cos_values = list(real_cosine_sims.values())
    real_mean = np.mean([abs(v) for v in real_cos_values])
    
    null_arr = np.array(null_distribution)
    null_mean = null_arr.mean()
    null_std = null_arr.std() + 1e-10
    
    # p-value: fraction of null that is <= real (we expect real < null)
    p_value = float(np.mean(null_arr <= real_mean))
    effect_size = float((null_mean - real_mean) / null_std)
    
    return {
        "null_distribution_mean": float(null_mean),
        "null_distribution_std": float(null_std),
        "real_mean_abs_cosine": float(real_mean),
        "p_value": p_value,
        "effect_size": effect_size,
        "n_permutations": n_permutations,
        "analysis_layer": analysis_layer,
    }


def main():
    parser = argparse.ArgumentParser(description="CI decomposition analysis")
    parser.add_argument("--activations-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/ci_decomposition")
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CI-Steering: CI Parameter Decomposition")
    print(f"Activations: {act_dir}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    ci_params = ["info_type", "recipient", "transmission_principle"]
    param_directions = {}

    for param_name in ci_params:
        ci_path = act_dir / f"ci_{param_name}"
        if not ci_path.exists():
            print(f"\n  Skipping {param_name} (not found at {ci_path})")
            continue

        print(f"\n" + "-" * 40)
        print(f"Analyzing: {param_name}")
        print("-" * 40)

        data = ActivationExtractor.load_activations(str(ci_path))
        activations = data["activations"]
        varied_values = data["varied_values"]

        unique_values = sorted(set(varied_values))
        print(f"  Values: {unique_values}")
        print(f"  Samples: {len(varied_values)}")

        results = compute_direction_for_parameter(activations, varied_values)
        param_directions[param_name] = results

        # Report top layers by probe accuracy
        sorted_layers = sorted(
            results.items(),
            key=lambda x: x[1]["probe_accuracy"],
            reverse=True,
        )
        print(f"\n  Top layers by probe accuracy:")
        for layer, info in sorted_layers[:5]:
            print(f"    Layer {layer:3d}: acc={info['probe_accuracy']:.4f}  "
                  f"var_explained={info['pca_variance_explained'][0]:.4f}")

    if len(param_directions) >= 2:
        print("\n" + "-" * 40)
        print("Cosine Similarity Between CI Parameter Directions")
        print("-" * 40)

        # Find a good layer (use the one with highest average probe accuracy)
        all_layers = set()
        for param_results in param_directions.values():
            all_layers.update(param_results.keys())

        best_layer = None
        best_avg_acc = 0
        for layer in all_layers:
            accs = []
            for param_results in param_directions.values():
                if layer in param_results:
                    accs.append(param_results[layer]["probe_accuracy"])
            if accs:
                avg = np.mean(accs)
                if avg > best_avg_acc:
                    best_avg_acc = avg
                    best_layer = layer

        print(f"\n  Analysis at layer {best_layer} (avg probe acc: {best_avg_acc:.4f})")

        sim_result = compute_cosine_similarity_matrix(param_directions, best_layer)
        param_names = sim_result["parameters"]
        sim_matrix = np.array(sim_result["similarity_matrix"])

        print(f"\n  Cosine similarity matrix:")
        # Header
        header = "  " + " " * 25 + "  ".join(f"{p:>20s}" for p in param_names)
        print(header)
        for i, name in enumerate(param_names):
            row = f"  {name:>25s}"
            for j in range(len(param_names)):
                row += f"  {sim_matrix[i, j]:>20.4f}"
            print(row)

        # Interpretation
        print(f"\n  Interpretation:")
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                sim = abs(sim_matrix[i, j])
                if sim < 0.3:
                    relation = "nearly orthogonal (separable)"
                elif sim < 0.6:
                    relation = "partially correlated"
                else:
                    relation = "highly correlated (entangled)"
                print(f"    {param_names[i]} vs {param_names[j]}: |cos|={sim:.4f} -> {relation}")

    perm_result = None
    if len(param_directions) >= 2:
        print("\n" + "-" * 40)
        print("Permutation Test for Direction Independence")
        print("-" * 40)
        
        # Collect real pairwise cosine similarities
        real_cosines = {}
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                real_cosines[(param_names[i], param_names[j])] = sim_matrix[i, j]
        
        # Load all activations for permutation test
        all_acts = {}
        all_vals = {}
        for param_name in ci_params:
            ci_path = act_dir / f"ci_{param_name}"
            if ci_path.exists():
                data = ActivationExtractor.load_activations(str(ci_path))
                all_acts[param_name] = data["activations"]
                all_vals[param_name] = data["varied_values"]
        
        perm_result = permutation_test_cosine_similarity(
            all_acts, all_vals, real_cosines, n_permutations=1000
        )
        
        print(f"  Real mean |cos|: {perm_result['real_mean_abs_cosine']:.4f}")
        print(f"  Null mean |cos|: {perm_result['null_distribution_mean']:.4f} ± {perm_result['null_distribution_std']:.4f}")
        print(f"  p-value (real < null): {perm_result['p_value']:.4f}")
        print(f"  Effect size: {perm_result['effect_size']:.2f}")
        
        if perm_result['p_value'] < 0.05:
            print("  → CI directions are significantly more independent than random partitions")
        else:
            print("  → CI directions are NOT significantly more independent than random")

    for param_name, results in param_directions.items():
        direction_dict = {}
        for layer_idx, info in results.items():
            direction_dict[layer_idx] = torch.from_numpy(info["direction"]).float()
        pt_path = out_dir / f"ci_{param_name}_directions.pt"
        torch.save(direction_dict, pt_path)
        print(f"  Saved direction vectors: {pt_path}")

    print(f"\n  Direction files can be loaded by:")
    print(f"    CICompositionalSteering.from_ci_directions_dir('{out_dir}')")
    print(f"    RepTuningConfig(ci_directions_dir='{out_dir}')")

    save_results = {}
    for param_name, results in param_directions.items():
        save_results[param_name] = {}
        for layer, info in results.items():
            save_results[param_name][str(layer)] = {
                k: v for k, v in info.items() if k != "direction"
            }

    if len(param_directions) >= 2:
        save_results["cosine_similarity"] = {
            "layer": sim_result["layer"],
            "parameters": sim_result["parameters"],
            "matrix": sim_result["similarity_matrix"],
        }

    if perm_result is not None:
        save_results["permutation_test"] = perm_result

    # Save summary of best layer per parameter
    summary = {}
    for param_name, results in param_directions.items():
        sorted_layers = sorted(results.items(), key=lambda x: x[1]["probe_accuracy"], reverse=True)
        best = sorted_layers[0]
        summary[param_name] = {
            "best_layer": best[0],
            "best_probe_accuracy": best[1]["probe_accuracy"],
            "pca_variance_explained_pc1": best[1]["pca_variance_explained"][0],
            "n_classes": best[1]["n_classes"],
            "class_names": best[1]["class_names"],
        }
    save_results["summary"] = summary

    with open(out_dir / "ci_decomposition_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")

        # Plot 1: Cosine similarity heatmap
        if len(param_directions) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                sim_matrix,
                xticklabels=[n.replace("_", " ").title() for n in param_names],
                yticklabels=[n.replace("_", " ").title() for n in param_names],
                annot=True,
                fmt=".3f",
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                ax=ax,
            )
            ax.set_title(f"Cosine Similarity of CI Parameter Directions (Layer {best_layer})", fontsize=13)
            fig.tight_layout()
            fig.savefig(out_dir / "ci_cosine_similarity.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Saved: ci_cosine_similarity.png")

        # Plot 2: Layer-wise probe accuracy for each CI parameter
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"info_type": "#3498db", "recipient": "#e67e22", "transmission_principle": "#2ecc71"}
        for param_name, results in param_directions.items():
            layers_sorted = sorted(results.keys())
            accs = [results[l]["probe_accuracy"] for l in layers_sorted]
            ax.plot(layers_sorted, accs, "o-", color=colors.get(param_name, "#999"),
                    linewidth=2, markersize=5,
                    label=param_name.replace("_", " ").title())
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Probe Accuracy", fontsize=12)
        ax.set_title("CI Parameter Separability Across Layers", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(out_dir / "ci_layer_probe_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: ci_layer_probe_accuracy.png")

        # Plot 3: PCA variance explained (PC1) per parameter across layers
        fig, ax = plt.subplots(figsize=(10, 5))
        for param_name, results in param_directions.items():
            layers_sorted = sorted(results.keys())
            var_pc1 = [results[l]["pca_variance_explained"][0] for l in layers_sorted]
            ax.plot(layers_sorted, var_pc1, "s--", color=colors.get(param_name, "#999"),
                    linewidth=2, markersize=5,
                    label=param_name.replace("_", " ").title())
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Variance Explained (PC1)", fontsize=12)
        ax.set_title("PCA Variance Explained by First Component", fontsize=13)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "ci_pca_variance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: ci_pca_variance.png")

    except ImportError:
        print("  matplotlib/seaborn not available, skipping visualization")

    print("\n" + "=" * 60)
    print("CI decomposition analysis complete!")
    print(f"Results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
