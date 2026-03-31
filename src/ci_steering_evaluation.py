#!/usr/bin/env python3

import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.model_utils import load_model, ModelHelper, resolve_api_key
from src.data.stimulus_generation import PrivacyStimulusGenerator
from src.reading.pca_reader import PCAReader
from src.reading.probe_reader import ProbeReader
from src.control.steering import PrivacySteering
from src.control.ci_steering import CICompositionalSteering
from src.evaluation.ci_eval import CIEvaluator


def run_method(
    label: str,
    helper: ModelHelper,
    prompts: list[str],
    steering_hook_fn,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Generate outputs for a given method configuration."""
    print(f"\n{'='*50}")
    print(f"Method: {label}")
    print(f"{'='*50}")
    outputs = helper.generate(
        texts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        steering_hook=steering_hook_fn,
    )
    return outputs


def evaluate_outputs(
    judge: CIEvaluator,
    outputs: list[str],
    scenarios: list[dict],
    baseline_results: dict = None,
) -> dict:
    """Run GPT-as-judge and compute metrics."""
    results = judge.evaluate_privacy_leakage(
        outputs=outputs,
        scenarios=scenarios,
    )
    summary = {
        "leakage_rate": results["overall_leakage_rate"],
        "refusal_rate": results["overall_refusal_rate"],
        "ncr": results["ci_norm_compliance_rate"],
    }
    if baseline_results is not None:
        ppi = judge.compute_ppi(baseline_results, results)
        summary["ppi"] = ppi["privacy_protection_improvement"]
    else:
        summary["ppi"] = 0.0

    print(f"  Leakage: {summary['leakage_rate']:.2%}  |  "
          f"Refusal: {summary['refusal_rate']:.2%}  |  "
          f"NCR: {summary['ncr']:.2%}  |  "
          f"PPI: {summary['ppi']:.2%}")

    return {**summary, "full_results": results}


def generate_comparison_plot(method_results: dict, out_dir: Path):
    """Generate a bar chart comparing all methods."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("  matplotlib/seaborn not available, skipping visualization")
        return

    methods = list(method_results.keys())
    leakage = [method_results[m]["leakage_rate"] * 100 for m in methods]
    ncr = [method_results[m]["ncr"] * 100 for m in methods]
    ppi = [method_results[m]["ppi"] * 100 for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Leakage rate (lower is better)
    colors = ["#e74c3c" if m == "No Steering" else
              "#f39c12" if "Standard" in m or "Baseline" in m else
              "#3498db" for m in methods]
    axes[0].barh(methods, leakage, color=colors, alpha=0.85)
    axes[0].set_xlabel("Leakage Rate (%)", fontsize=12)
    axes[0].set_title("Leakage Rate (lower is better)", fontsize=13)
    axes[0].set_xlim(0, 105)
    for i, v in enumerate(leakage):
        axes[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10)

    # NCR (higher is better)
    axes[1].barh(methods, ncr, color=colors, alpha=0.85)
    axes[1].set_xlabel("Norm Compliance Rate (%)", fontsize=12)
    axes[1].set_title("CI Norm Compliance (higher is better)", fontsize=13)
    axes[1].set_xlim(0, 105)
    for i, v in enumerate(ncr):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10)

    # PPI (higher is better)
    axes[2].barh(methods, ppi, color=colors, alpha=0.85)
    axes[2].set_xlabel("Privacy Protection Improvement (%)", fontsize=12)
    axes[2].set_title("PPI (higher is better)", fontsize=13)
    for i, v in enumerate(ppi):
        axes[2].text(v + 1 if v >= 0 else v - 8, i, f"{v:.1f}%",
                     va="center", fontsize=10)

    fig.suptitle("CI-Decomposed Steering: Method Comparison", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "ci_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: ci_method_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="CI-Decomposed Steering Comparative Evaluation"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--reader-dir", type=str, required=True,
                        help="Path to fitted reader (PCA or probe)")
    parser.add_argument("--reader-type", type=str, default="probe",
                        choices=["pca", "probe"])
    parser.add_argument("--ci-dir", type=str, required=True,
                        help="Path to CI directions from ci_decomposition.py "
                        "(should contain ci_*_directions.pt files)")
    parser.add_argument("--stimuli-dir", type=str, default="data/stimuli")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/ci_steering_comparison")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Single alpha (overrides --alphas)")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                        help="Alpha values to sweep (default: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])")
    parser.add_argument("--top-k-layers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of evaluation scenarios")
    parser.add_argument("--openai-api-key", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_values = [args.alpha] if args.alpha is not None else args.alphas

    print("=" * 60)
    print("CI-Decomposed Steering: Comparative Evaluation")
    print(f"Model:       {args.model}")
    print(f"Reader:      {args.reader_dir} ({args.reader_type})")
    print(f"CI dirs:     {args.ci_dir}")
    print(f"Alphas:      {alpha_values}")
    print(f"Top-k:       {args.top_k_layers}")
    print("=" * 60)

    print("\nLoading model...")
    model, tokenizer = load_model(args.model, dtype="float16")
    helper = ModelHelper(model, tokenizer)

    print("Loading reader...")
    if args.reader_type == "probe":
        reader = ProbeReader()
    else:
        reader = PCAReader()
    reader.load(args.reader_dir)

    print("Loading evaluation stimuli...")
    balanced_path = Path(args.stimuli_dir) / "function_stimuli_balanced.json"
    if balanced_path.exists():
        func_stimuli = PrivacyStimulusGenerator.load(str(balanced_path))
    else:
        gen = PrivacyStimulusGenerator(seed=42)
        func_stimuli = gen.generate_function_stimuli_balanced(
            num_inappropriate=100, num_appropriate=100
        )
    eval_stimuli = func_stimuli[:args.n_eval]
    eval_prompts = [
        PrivacyStimulusGenerator.wrap_function_template(s, tokenizer=tokenizer)
        for s in eval_stimuli
    ]
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
    print(f"  Evaluation scenarios: {len(eval_prompts)}")

    api_key = resolve_api_key(args.openai_api_key)
    if not api_key:
        print("ERROR: No OpenAI API key. Set OPENAI_API_KEY or create .api_key file.")
        sys.exit(1)

    judge = CIEvaluator(
        api_key=api_key,
        model="gpt-4o-mini",
        cache_dir=str(out_dir / "judge_cache"),
    )

    baseline_outputs = run_method(
        "No Steering",
        helper, eval_prompts,
        steering_hook_fn=None,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    baseline_eval = evaluate_outputs(judge, baseline_outputs, scenarios)
    baseline_full = baseline_eval["full_results"]

    all_alpha_summaries = {}

    for alpha in alpha_values:
        print(f"\n{'#'*60}")
        print(f"  Alpha = {alpha}")
        print(f"{'#'*60}")

        if args.reader_type == "probe":
            std_steerer = PrivacySteering.from_probe_reader(
                model_helper=helper,
                probe_reader=reader,
                alpha=alpha,
                top_k_layers=args.top_k_layers,
            )
        else:
            std_steerer = PrivacySteering.from_pca_reader(
                model_helper=helper,
                pca_reader=reader,
                alpha=alpha,
                top_k_layers=args.top_k_layers,
            )
        std_outputs = run_method(
            f"Standard Steering (α={alpha})",
            helper, eval_prompts,
            steering_hook_fn=std_steerer._make_steering_hook(),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        std_eval = evaluate_outputs(judge, std_outputs, scenarios, baseline_full)

        if args.reader_type == "probe":
            pw_steerer = PrivacySteering.from_probe_reader_weighted(
                model_helper=helper,
                probe_reader=reader,
                alpha=alpha,
                top_k_layers=args.top_k_layers,
            )
        else:
            best_layers = reader.get_best_layers(top_k=args.top_k_layers)
            directions = {
                layer: -reader.get_privacy_vector(layer)
                for layer in best_layers
            }
            layer_weights = {}
            if hasattr(reader, "layer_scores"):
                for layer in best_layers:
                    scores = reader.layer_scores.get(layer, {})
                    layer_weights[layer] = scores.get(
                        "train_auroc", scores.get("auroc", 0.5)
                    )
            pw_steerer = PrivacySteering(
                model_helper=helper,
                privacy_directions=directions,
                alpha=alpha,
                steering_layers=best_layers,
                layer_weights=layer_weights,
            )

        pw_outputs = run_method(
            f"Probe-Weighted Steering (α={alpha})",
            helper, eval_prompts,
            steering_hook_fn=pw_steerer._make_steering_hook(),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        pw_eval = evaluate_outputs(judge, pw_outputs, scenarios, baseline_full)

        ci_steerer = CICompositionalSteering.from_ci_directions_dir(
            model_helper=helper,
            ci_dir=args.ci_dir,
            alphas={"info_type": alpha, "recipient": alpha,
                    "transmission_principle": alpha},
            top_k_layers=args.top_k_layers,
        )
        print(f"\n  CI Steerer config:\n{ci_steerer.describe()}")

        ci_outputs = run_method(
            f"CI-Decomposed all (α={alpha})",
            helper, eval_prompts,
            steering_hook_fn=ci_steerer._make_steering_hook(),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        ci_eval = evaluate_outputs(judge, ci_outputs, scenarios, baseline_full)

        ablation_results = {}
        for param_name in CICompositionalSteering.CI_PARAMETERS:
            if param_name not in ci_steerer.ci_directions:
                print(f"\n  Skipping ablation for {param_name} (no directions)")
                continue

            ablation_alphas = {p: 0.0 for p in ci_steerer.ci_directions}
            ablation_alphas[param_name] = alpha

            ci_steerer.alphas = ablation_alphas
            label = f"CI: {param_name} only (α={alpha})"
            abl_outputs = run_method(
                label,
                helper, eval_prompts,
                steering_hook_fn=ci_steerer._make_steering_hook(),
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )
            abl_eval = evaluate_outputs(judge, abl_outputs, scenarios, baseline_full)
            ablation_results[param_name] = {
                "leakage_rate": abl_eval["leakage_rate"],
                "refusal_rate": abl_eval["refusal_rate"],
                "ncr": abl_eval["ncr"],
                "ppi": abl_eval["ppi"],
                "outputs": abl_outputs[:5],
            }

        ci_steerer.alphas = {
            p: alpha for p in ci_steerer.ci_directions
        }

        method_summary = {
            "No Steering": {
                "leakage_rate": baseline_eval["leakage_rate"],
                "refusal_rate": baseline_eval["refusal_rate"],
                "ncr": baseline_eval["ncr"],
                "ppi": 0.0,
            },
            "Standard Steering": {
                "leakage_rate": std_eval["leakage_rate"],
                "refusal_rate": std_eval["refusal_rate"],
                "ncr": std_eval["ncr"],
                "ppi": std_eval["ppi"],
            },
            "Probe-Weighted Steering": {
                "leakage_rate": pw_eval["leakage_rate"],
                "refusal_rate": pw_eval["refusal_rate"],
                "ncr": pw_eval["ncr"],
                "ppi": pw_eval["ppi"],
            },
            "CI-Decomposed (all)": {
                "leakage_rate": ci_eval["leakage_rate"],
                "refusal_rate": ci_eval["refusal_rate"],
                "ncr": ci_eval["ncr"],
                "ppi": ci_eval["ppi"],
            },
        }
        for param_name, abl in ablation_results.items():
            method_summary[f"CI: {param_name}"] = {
                "leakage_rate": abl["leakage_rate"],
                "refusal_rate": abl["refusal_rate"],
                "ncr": abl["ncr"],
                "ppi": abl["ppi"],
            }

        all_alpha_summaries[str(alpha)] = {
            "methods": method_summary,
            "ablation_details": ablation_results,
        }

        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY (α={alpha})")
        print(f"{'='*80}")
        print(f"{'Method':<30s} {'Leak%':>8s} {'Refuse%':>8s} {'NCR%':>8s} {'PPI%':>8s}")
        print("-" * 80)
        for method, metrics in method_summary.items():
            print(f"{method:<30s} "
                  f"{metrics['leakage_rate']*100:>7.1f}% "
                  f"{metrics['refusal_rate']*100:>7.1f}% "
                  f"{metrics['ncr']*100:>7.1f}% "
                  f"{metrics['ppi']*100:>7.1f}%")
        print("=" * 80)

        generate_comparison_plot(method_summary, out_dir)

    full_results = {
        "model": args.model,
        "alphas": alpha_values,
        "reader_type": args.reader_type,
        "reader_dir": args.reader_dir,
        "ci_dir": args.ci_dir,
        "n_eval": len(eval_prompts),
        "top_k_layers": args.top_k_layers,
        "results_by_alpha": all_alpha_summaries,
    }

    with open(out_dir / "ci_comparison_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n  Saved: ci_comparison_results.json")

    print(f"\nResults saved to: {out_dir}")

    del model, helper
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
