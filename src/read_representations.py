#!/usr/bin/env python3
"""Representation reading via PCA and probes."""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.activation_extractor import ActivationExtractor
from src.reading.pca_reader import PCAReader
from src.reading.probe_reader import ProbeReader


def main():
    parser = argparse.ArgumentParser(description="Representation reading for CI-Steering")
    parser.add_argument("--activations-dir", type=str, required=True,
                        help="Directory with extracted activations (e.g., outputs/activations/Llama-3.1-8B-Instruct)")
    parser.add_argument("--output-dir", type=str, default="outputs/reading")
    parser.add_argument("--n-components", type=int, default=10,
                        help="Number of PCA components")
    parser.add_argument("--method", type=str, default="both",
                        choices=["pca", "probe", "both"])
    parser.add_argument("--visualize", action="store_true",
                        help="Generate t-SNE / UMAP visualizations")
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CI-Steering Representation Reading")
    print(f"Activations: {act_dir}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # ---- Load activations ----
    print("\nLoading concept activations...")
    train_data = ActivationExtractor.load_activations(str(act_dir / "concept_train"))
    test_data = ActivationExtractor.load_activations(str(act_dir / "concept_test"))

    train_acts = train_data["activations"]
    train_labels = train_data["labels"]
    train_pair_ids = train_data.get("pair_ids", None)
    test_acts = test_data["activations"]
    test_labels = test_data["labels"]

    layers = sorted(train_acts.keys())
    print(f"  Layers: {len(layers)} (indices: {layers[0]}..{layers[-1]})")
    print(f"  Train: {len(train_labels)} samples, Test: {len(test_labels)} samples")
    if train_pair_ids is not None:
        n_pairs = len(set(train_pair_ids))
        print(f"  Pairs: {n_pairs} (will use paired-difference PCA per RepE methodology)")

    all_results = {}

    # ---- PCA-based reading (RQ1) ----
    if args.method in ("pca", "both"):
        print("\n" + "-" * 40)
        print("PCA-Based Reading (Unsupervised)")
        print("-" * 40)

        pca_reader = PCAReader(n_components=args.n_components)
        pca_reader.fit(train_acts, train_labels, pair_ids=train_pair_ids, method="pca")

        # Evaluate on test set
        pca_eval = pca_reader.evaluate(test_acts, test_labels)

        # Report layer-wise results
        print("\n  Layer-wise test accuracy (top 10):")
        sorted_layers = sorted(pca_eval.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for layer, metrics in sorted_layers[:10]:
            print(f"    Layer {layer:3d}: acc={metrics['accuracy']:.4f}  auroc={metrics['auroc']:.4f}")

        best_layers = pca_reader.get_best_layers(top_k=5, metric="auroc")
        print(f"\n  Best layers: {best_layers}")

        # Variance explained
        var_explained = pca_reader.get_variance_explained()
        if var_explained:
            best_layer = best_layers[0]
            if best_layer in var_explained:
                print(f"  Variance explained (layer {best_layer}):")
                for i, v in enumerate(var_explained[best_layer][:5]):
                    print(f"    PC{i+1}: {v:.4f}")

        # Save
        pca_reader.save(str(out_dir / "pca_reader"))
        all_results["pca"] = {
            str(k): v for k, v in pca_eval.items()
        }

    # ---- Probe-based reading (RQ1 + RQ2) ----
    if args.method in ("probe", "both"):
        print("\n" + "-" * 40)
        print("Probe-Based Reading (Supervised)")
        print("-" * 40)

        probe_reader = ProbeReader(max_iter=1000, C=1.0, cv_folds=5)
        probe_reader.fit(train_acts, train_labels)

        # Evaluate on test set
        probe_eval = probe_reader.evaluate(test_acts, test_labels)

        # Report layer-wise results
        print("\n  Layer-wise test accuracy (top 10):")
        sorted_layers = sorted(probe_eval.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for layer, metrics in sorted_layers[:10]:
            print(f"    Layer {layer:3d}: acc={metrics['accuracy']:.4f}  auroc={metrics['auroc']:.4f}")

        # Cross-validation scores
        print("\n  Cross-validation scores (top 5):")
        best_layers = probe_reader.get_best_layers(top_k=5, metric="cv_mean")
        for layer in best_layers:
            scores = probe_reader.layer_scores[layer]
            print(f"    Layer {layer:3d}: cv_mean={scores['cv_mean']:.4f} +/- {scores['cv_std']:.4f}")

        # Save
        probe_reader.save(str(out_dir / "probe_reader"))
        all_results["probe"] = {
            str(k): {kk: vv for kk, vv in v.items() if kk != "predictions" and kk != "probabilities"}
            for k, v in probe_eval.items()
        }

    # ---- Cross-task evaluation: apply concept probe to behavioral activations ----
    if args.method in ("probe", "both"):
        func_act_dir = act_dir / "function"
        if func_act_dir.exists():
            print("\n" + "-" * 40)
            print("Cross-Task Evaluation (Concept Probe → Behavioral Activations)")
            print("-" * 40)

            func_data = ActivationExtractor.load_activations(str(func_act_dir))
            func_acts = func_data["activations"]

            # Use is_appropriate labels if available (balanced stimuli),
            # otherwise fall back to all-inappropriate (legacy format).
            if "labels" in func_data and func_data["labels"] is not None:
                func_labels = func_data["labels"]
                n_app = sum(1 for l in func_labels if l)
                n_inapp = sum(1 for l in func_labels if not l)
                print(f"  Behavioral samples: {len(func_labels)} "
                      f"({n_app} appropriate, {n_inapp} inappropriate)")
            else:
                func_labels = [False] * next(iter(func_acts.values())).shape[0]
                print(f"  Behavioral samples: {len(func_labels)} (all label=inappropriate)")

            cross_eval = probe_reader.evaluate_cross_task(func_acts, func_labels)

            print("\n  Cross-task accuracy (top 10 layers):")
            sorted_cross = sorted(cross_eval.items(), key=lambda x: x[1]["accuracy"], reverse=True)
            for layer, metrics in sorted_cross[:10]:
                print(f"    Layer {layer:3d}: acc={metrics['accuracy']:.4f}  auroc={metrics['auroc']:.4f}")

            best_cross_layer = sorted_cross[0][0]
            best_cross_acc = sorted_cross[0][1]["accuracy"]
            print(f"\n  Best cross-task layer: {best_cross_layer} (acc={best_cross_acc:.4f})")

            all_results["cross_task"] = {
                str(k): {kk: vv for kk, vv in v.items() if kk not in ("predictions", "probabilities")}
                for k, v in cross_eval.items()
            }
        else:
            print("\n  Skipping cross-task evaluation (no function activations found)")

    # ---- Multi-component PCA analysis (multi-dimensionality evidence) ----
    if args.method in ("pca", "both") and "pca" in all_results:
        print("\n" + "-" * 40)
        print("Multi-Component PCA Analysis")
        print("-" * 40)

        k_values = [1, 3, 5, 10]
        multi_comp = pca_reader.evaluate_multi_component(test_acts, test_labels, k_values=k_values)

        if multi_comp:
            # Report best layer for each k
            for k in k_values:
                best_layer_k = max(multi_comp.keys(), key=lambda l: multi_comp[l].get(k, {}).get("accuracy", 0))
                best_acc_k = multi_comp[best_layer_k].get(k, {}).get("accuracy", 0)
                best_auroc_k = multi_comp[best_layer_k].get(k, {}).get("auroc", 0)
                print(f"  PCA-{k:2d}: best acc={best_acc_k:.4f}  auroc={best_auroc_k:.4f}  (layer {best_layer_k})")

            all_results["multi_component_pca"] = {
                str(layer): {str(k): v for k, v in layer_data.items()}
                for layer, layer_data in multi_comp.items()
            }

    # ---- Generate layer-wise accuracy plot data ----
    print("\n" + "-" * 40)
    print("Layer-wise Analysis Summary")
    print("-" * 40)

    layer_summary = {}
    for layer in layers:
        entry = {"layer": layer}
        if "pca" in all_results:
            pca_key = str(layer)
            if pca_key in all_results["pca"]:
                entry["pca_accuracy"] = all_results["pca"][pca_key]["accuracy"]
                entry["pca_auroc"] = all_results["pca"][pca_key]["auroc"]
        if "probe" in all_results:
            probe_key = str(layer)
            if probe_key in all_results["probe"]:
                entry["probe_accuracy"] = all_results["probe"][probe_key]["accuracy"]
                entry["probe_auroc"] = all_results["probe"][probe_key]["auroc"]
        layer_summary[layer] = entry

    with open(out_dir / "layer_summary.json", "w") as f:
        json.dump(layer_summary, f, indent=2)

    # ---- Visualization ----
    print("\n" + "-" * 40)
    print("Generating Visualizations")
    print("-" * 40)

    _generate_layer_plot(layer_summary, out_dir)

    if args.visualize:
        _generate_visualizations(test_acts, test_labels, test_data, best_layers, out_dir)

    # ---- Save all results ----
    with open(out_dir / "reading_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Representation reading complete!")
    print(f"Results saved to: {out_dir}")
    print("=" * 60)


def _generate_layer_plot(layer_summary, out_dir):
    """Generate a layer-wise accuracy/AUROC plot for PCA and probe readers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("  matplotlib/seaborn not available, skipping layer plot")
        return

    layers_sorted = sorted(layer_summary.keys())
    has_pca = any("pca_auroc" in layer_summary[l] for l in layers_sorted)
    has_probe = any("probe_auroc" in layer_summary[l] for l in layers_sorted)

    if not has_pca and not has_probe:
        print("  No data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Plot AUROC
    ax = axes[0]
    if has_pca:
        y = [layer_summary[l].get("pca_auroc", None) for l in layers_sorted]
        ax.plot(layers_sorted, y, "o-", color="#3498db", linewidth=2, markersize=5, label="PCA (unsupervised)")
    if has_probe:
        y = [layer_summary[l].get("probe_auroc", None) for l in layers_sorted]
        ax.plot(layers_sorted, y, "s-", color="#e67e22", linewidth=2, markersize=5, label="Probe (supervised)")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Layer-wise AUROC", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.35, 1.05)

    # Plot accuracy
    ax = axes[1]
    if has_pca:
        y = [layer_summary[l].get("pca_accuracy", None) for l in layers_sorted]
        ax.plot(layers_sorted, y, "o-", color="#3498db", linewidth=2, markersize=5, label="PCA (unsupervised)")
    if has_probe:
        y = [layer_summary[l].get("probe_accuracy", None) for l in layers_sorted]
        ax.plot(layers_sorted, y, "s-", color="#e67e22", linewidth=2, markersize=5, label="Probe (supervised)")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Layer-wise Accuracy", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.35, 1.05)

    fig.suptitle("Privacy Representation Reading: Layer Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "layer_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: layer_analysis.png")


def _generate_visualizations(test_acts, test_labels, test_data, best_layers, out_dir):
    """Generate t-SNE or UMAP visualizations of privacy representations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not available, skipping visualizations")
        return

    labels_arr = np.array(test_labels)
    info_types = test_data.get("info_types", [])

    for layer in best_layers[:3]:  # Top 3 layers
        if layer not in test_acts:
            continue

        acts = test_acts[layer].float().numpy()

        # Try UMAP first, fall back to t-SNE
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            method = "UMAP"
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            method = "t-SNE"

        print(f"  Computing {method} for layer {layer}...")
        embedding = reducer.fit_transform(acts)

        # Plot 1: Colored by appropriateness
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = ["#e74c3c" if not label else "#2ecc71" for label in labels_arr]
        axes[0].scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.6, s=20)
        axes[0].set_title(f"Layer {layer} - Privacy Appropriateness")
        axes[0].legend(
            handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', label='Appropriate'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', label='Inappropriate'),
            ],
            loc="best",
        )

        # Plot 2: Colored by information type
        if info_types:
            unique_types = sorted(set(info_types))
            cmap = plt.cm.get_cmap("tab10", len(unique_types))
            type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}
            colors_by_type = [type_to_color[t] for t in info_types]
            axes[1].scatter(embedding[:, 0], embedding[:, 1], c=colors_by_type, alpha=0.6, s=20)
            axes[1].set_title(f"Layer {layer} - Information Type")
            axes[1].legend(
                handles=[
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=type_to_color[t], label=t)
                    for t in unique_types
                ],
                loc="best",
                fontsize=7,
            )

        plt.tight_layout()
        fig.savefig(out_dir / f"viz_layer{layer}_{method.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: viz_layer{layer}_{method.lower()}.png")


if __name__ == "__main__":
    main()
