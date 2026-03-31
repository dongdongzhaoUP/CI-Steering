#!/usr/bin/env python3

import argparse
import json
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.model_utils import load_model, ModelHelper, resolve_api_key
from src.data.stimulus_generation import PrivacyStimulusGenerator
from src.reading.pca_reader import PCAReader
from src.reading.probe_reader import ProbeReader
from src.control.steering import PrivacySteering
from src.evaluation.ci_eval import CIEvaluator


def generate_visualizations(results: dict, out_dir: Path):
    """Generate plots for steering results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("  matplotlib/seaborn not available, skipping visualizations")
        return

    steering = results.get("steering_results", {})
    if not steering:
        return

    alphas = [0.0]
    leakage = [results["unsteered"].get("overall_leakage_rate", 0)]
    refusal = [results["unsteered"].get("overall_refusal_rate", 0)]

    for a_str, data in sorted(steering.items(), key=lambda x: float(x[0])):
        alphas.append(float(a_str))
        sr = data.get("steered_results", {})
        leakage.append(sr.get("overall_leakage_rate", 0))
        refusal.append(sr.get("overall_refusal_rate", 0))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_leak = "#e74c3c"
    color_ref = "#2ecc71"

    ax1.plot(alphas, [l * 100 for l in leakage], "o-", color=color_leak, linewidth=2, markersize=8, label="Leakage rate")
    ax1.set_xlabel("Steering strength (α)", fontsize=13)
    ax1.set_ylabel("Leakage rate (%)", fontsize=13, color=color_leak)
    ax1.tick_params(axis="y", labelcolor=color_leak)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    ax2.plot(alphas, [r * 100 for r in refusal], "s--", color=color_ref, linewidth=2, markersize=8, label="Refusal rate")
    ax2.set_ylabel("Refusal rate (%)", fontsize=13, color=color_ref)
    ax2.tick_params(axis="y", labelcolor=color_ref)
    ax2.set_ylim(-5, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)
    ax1.set_title("Privacy Steering: Leakage & Refusal vs α (GPT Judge)", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_dir / "steering_leakage_refusal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: steering_leakage_refusal.png")

    breakdown = results["unsteered"].get("per_type_breakdown", {})
    if breakdown:
        types = sorted(breakdown.keys())
        leak_vals = [breakdown[t]["leakage_rate"] * 100 for t in types]
        ref_vals = [breakdown[t]["refusal_rate"] * 100 for t in types]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(types))
        w = 0.35
        ax.bar(x - w / 2, leak_vals, w, label="Leakage", color=color_leak, alpha=0.8)
        ax.bar(x + w / 2, ref_vals, w, label="Refusal", color=color_ref, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Rate (%)", fontsize=12)
        ax.set_title("Unsteered: Per-Type Leakage & Refusal", fontsize=14)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 105)

        fig.tight_layout()
        fig.savefig(out_dir / "per_type_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: per_type_breakdown.png")

    reader_info = results.get("reader_layer_scores", {})
    if reader_info:
        layers = sorted(reader_info.keys(), key=int)
        accs = []
        for l in layers:
            v = reader_info[l]
            if isinstance(v, dict):
                accs.append(v.get("auroc", v.get("train_auroc", v.get("cv_mean", 0.5))))
            else:
                accs.append(v)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([int(l) for l in layers], accs, "o-", color="#3498db", linewidth=2, markersize=5)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("PCA Reading AUROC", fontsize=13)
        ax.set_title("Layer-wise Privacy Direction Quality", fontsize=14)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
        ax.legend()

        fig.tight_layout()
        fig.savefig(out_dir / "layer_pca_auroc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: layer_pca_auroc.png")


def main():
    parser = argparse.ArgumentParser(description="Privacy steering and evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--reader-dir", type=str, required=True,
                        help="Directory with fitted reader (e.g., outputs/reading/pca_reader)")
    parser.add_argument("--reader-type", type=str, default="pca",
                        choices=["pca", "probe"])
    parser.add_argument("--stimuli-dir", type=str, default="data/stimuli")
    parser.add_argument("--output-dir", type=str, default="outputs/steering")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Single alpha value (overrides --alphas)")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                        help="Alpha values for sweep (default: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])")
    parser.add_argument("--top-k-layers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="OpenAI API key for GPT judge (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["model"]["name"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CI-Steering: Steering & Evaluation")
    print(f"Model: {model_name}")
    print(f"Reader: {args.reader_dir} ({args.reader_type})")
    print("=" * 60)

    # ---- Load model ----
    print("\nLoading model...")
    model, tokenizer = load_model(model_name, dtype=config["model"].get("dtype", "float16"))
    helper = ModelHelper(model, tokenizer)

    # ---- Load reader ----
    print("Loading privacy reader...")
    if args.reader_type == "pca":
        reader = PCAReader()
        reader.load(args.reader_dir)
    else:
        reader = ProbeReader()
        reader.load(args.reader_dir)

    # ---- Determine alpha values ----
    if args.alpha is not None:
        alpha_values = [args.alpha]
    else:
        alpha_values = args.alphas

    print(f"Alpha values: {alpha_values}")

    # ---- Load evaluation stimuli ----
    print("\nLoading evaluation stimuli...")
    balanced_path = Path(args.stimuli_dir) / "function_stimuli_balanced.json"
    if balanced_path.exists():
        func_stimuli = PrivacyStimulusGenerator.load(str(balanced_path))
    else:
        gen = PrivacyStimulusGenerator(seed=42)
        func_stimuli = gen.generate_function_stimuli_balanced(
            num_inappropriate=100, num_appropriate=100
        )

    # Use 200 balanced samples (100 inappropriate + 100 appropriate)
    eval_stimuli = func_stimuli[:200]

    # Wrap stimuli (handles both new roleplay dict and legacy string formats)
    eval_prompts = [
        PrivacyStimulusGenerator.wrap_function_template(s)
        for s in eval_stimuli
    ]
    info_types = [s["info_type"] for s in eval_stimuli]
    print(f"  Evaluation scenarios: {len(eval_prompts)}")

    stimuli_log = []
    for i, s in enumerate(eval_stimuli):
        stimuli_log.append({
            "index": i,
            "info_type": s["info_type"],
            "person": s.get("person", ""),
            "confidant": s.get("confidant", ""),
            "questioner": s.get("questioner", ""),
            "private_info": s.get("private_info", ""),
            "context": s.get("context", ""),
            "secret_keywords": s.get("secret_keywords", []),
            "prompt": eval_prompts[i],
        })

    print("\n" + "-" * 40)
    print("Generating UNSTEERED outputs (baseline)...")
    print("-" * 40)

    unsteered_outputs = helper.generate(
        texts=eval_prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # GPT-as-judge evaluation
    print("\n  Running GPT-as-judge evaluation (unsteered)...")
    api_key = resolve_api_key(args.openai_api_key)
    if not api_key:
        print("ERROR: No OpenAI API key. Set OPENAI_API_KEY or create .api_key file.")
        sys.exit(1)

    judge = CIEvaluator(
        api_key=api_key,
        model="gpt-4o-mini",
        cache_dir=str(out_dir / "judge_cache"),
    )
    scenarios = [
        {
            "subject": s.get("person", "someone"),
            "info_type": s.get("info_type", "personal info"),
            "recipient": s.get("questioner", "a third party"),
            "context": s.get("context", "social interaction"),
            "private_info": s.get("private_info", ""),
        }
        for s in eval_stimuli
    ]
    unsteered_results = judge.evaluate_privacy_leakage(
        outputs=unsteered_outputs,
        scenarios=scenarios,
    )
    print(f"  GPT-judge evaluation:")
    print(f"    Leakage rate: {unsteered_results['overall_leakage_rate']:.2%}")
    print(f"    Refusal rate: {unsteered_results['overall_refusal_rate']:.2%}")
    print(f"    CI Norm Compliance: {unsteered_results['ci_norm_compliance_rate']:.2%}")

    # Save per-sample unsteered data
    per_sample_unsteered = []
    per_sample_eval = unsteered_results.get("per_sample", [{}] * len(unsteered_outputs))
    for i, (prompt, output) in enumerate(zip(eval_prompts, unsteered_outputs)):
        sample = per_sample_eval[i] if i < len(per_sample_eval) else {}
        per_sample_unsteered.append({
            "index": i,
            "info_type": info_types[i],
            "person": eval_stimuli[i].get("person", ""),
            "private_info": eval_stimuli[i].get("private_info", ""),
            "prompt": prompt[:500],
            "response": output,
            "leaked": sample.get("leaked", None),
            "refused": sample.get("refused", None),
            "appropriate": sample.get("appropriate", None),
            "confidence": sample.get("confidence", None),
            "reasoning": sample.get("reasoning", ""),
        })

    # ---- Steered generation (sweep alpha) ----
    all_steering_results = {}
    per_sample_steered = {}

    for alpha in alpha_values:
        print(f"\n" + "-" * 40)
        print(f"Generating STEERED outputs (alpha={alpha})...")
        print("-" * 40)

        if args.reader_type == "pca":
            steerer = PrivacySteering.from_pca_reader(
                model_helper=helper,
                pca_reader=reader,
                alpha=alpha,
                top_k_layers=args.top_k_layers,
            )
        else:
            steerer = PrivacySteering.from_probe_reader(
                model_helper=helper,
                probe_reader=reader,
                alpha=alpha,
                top_k_layers=args.top_k_layers,
            )

        steered_outputs = steerer.generate(
            prompts=eval_prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

        # GPT-as-judge evaluation
        print(f"\n  Running GPT-as-judge evaluation (alpha={alpha})...")
        steered_gpt_results = judge.evaluate_privacy_leakage(
            outputs=steered_outputs,
            scenarios=scenarios,
        )
        print(f"  GPT-judge evaluation:")
        print(f"    Leakage rate: {steered_gpt_results['overall_leakage_rate']:.2%}")
        print(f"    Refusal rate: {steered_gpt_results['overall_refusal_rate']:.2%}")
        print(f"    CI Norm Compliance: {steered_gpt_results['ci_norm_compliance_rate']:.2%}")

        ppi = judge.compute_ppi(unsteered_results, steered_gpt_results)
        print(f"    Privacy Protection Improvement: {ppi['privacy_protection_improvement']:.2%}")

        alpha_result = {
            "steered_results": {
                k: v for k, v in steered_gpt_results.items() if k != "per_sample"
            },
            "ppi": ppi,
            "sample_outputs": steered_outputs[:5],
        }

        all_steering_results[alpha] = alpha_result

        # Save per-sample steered data
        steered_per_sample = steered_gpt_results.get("per_sample", [{}] * len(steered_outputs))
        per_sample_steered[str(alpha)] = []
        for i, output in enumerate(steered_outputs):
            sample = steered_per_sample[i] if i < len(steered_per_sample) else {}
            per_sample_steered[str(alpha)].append({
                "index": i,
                "info_type": info_types[i],
                "response": output,
                "leaked": sample.get("leaked", None),
                "refused": sample.get("refused", None),
                "appropriate": sample.get("appropriate", None),
                "reasoning": sample.get("reasoning", ""),
            })

    # ---- Save results ----
    best_alpha = min(
        all_steering_results.keys(),
        key=lambda a: all_steering_results[a]["steered_results"]["overall_leakage_rate"]
    )
    print(f"\n  Best alpha (lowest leakage): {best_alpha}")

    reader_scores = {str(k): v for k, v in reader.layer_scores.items()} if hasattr(reader, 'layer_scores') else {}

    final_results = {
        "model": model_name,
        "reader_type": args.reader_type,
        "reader_dir": args.reader_dir,
        "eval_method": "gpt_judge",
        "judge_model": "gpt-4o-mini",
        "n_eval_scenarios": len(eval_stimuli),
        "alpha_values": alpha_values,
        "top_k_layers": args.top_k_layers,
        "reader_layer_scores": reader_scores,
        "unsteered": {
            k: v for k, v in unsteered_results.items() if k != "per_sample"
        },
        "steering_results": {
            str(k): v for k, v in all_steering_results.items()
        },
        "best_alpha": best_alpha,
    }

    with open(out_dir / "steering_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    # Save detailed per-sample data (prompts, responses, evaluations)
    detailed = {
        "stimuli": stimuli_log,
        "unsteered_samples": per_sample_unsteered,
        "steered_samples": per_sample_steered,
    }
    with open(out_dir / "steering_samples.json", "w") as f:
        json.dump(detailed, f, indent=2, default=str)

    print(f"\n  Saved: steering_results.json")
    print(f"  Saved: steering_samples.json (prompts + responses)")

    # ---- Generate visualizations ----
    print("\n" + "-" * 40)
    print("Generating visualizations...")
    print("-" * 40)
    generate_visualizations(final_results, out_dir)

    print("\n" + "=" * 60)
    print("Steering evaluation complete!")
    print(f"Results saved to: {out_dir}")
    print("=" * 60)

    del model, helper
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
