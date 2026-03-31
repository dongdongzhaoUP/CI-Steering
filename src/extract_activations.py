#!/usr/bin/env python3
"""
Extract hidden-state activations from LLMs for privacy stimuli.

Runs the LAT pipeline:
  1. Load generated stimuli
  2. Wrap in LAT templates
  3. Forward-pass through the model with hooks
  4. Save per-layer activations
"""

import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.model_utils import load_model, ModelHelper
from src.data.stimulus_generation import PrivacyStimulusGenerator
from src.extraction.activation_extractor import ActivationExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract activations for CI-Steering")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--stimuli-dir", type=str, default="data/stimuli")
    parser.add_argument("--output-dir", type=str, default="outputs/activations")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-position", type=str, default="last",
                        choices=["last", "concept"])
    parser.add_argument("--skip-concept", action="store_true")
    parser.add_argument("--skip-function", action="store_true")
    parser.add_argument("--skip-ci", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["model"]["name"]
    dtype = config["model"].get("dtype", "float16")

    # Create a safe directory name from the model name
    model_short = model_name.split("/")[-1]
    output_dir = Path(args.output_dir) / model_short

    print("=" * 60)
    print(f"CI-Steering Activation Extraction")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(model_name, dtype=dtype)
    helper = ModelHelper(model, tokenizer)
    print(f"  Layers: {helper.num_layers}, Hidden size: {helper.hidden_size}")

    # Create extractor
    extractor = ActivationExtractor(
        model_helper=helper,
        output_dir=str(output_dir),
        token_position=args.token_position,
        batch_size=args.batch_size,
    )

    stimuli_dir = Path(args.stimuli_dir)

    # --- Concept-level activations (RQ1) ---
    if not args.skip_concept:
        print("\n" + "-" * 40)
        print("Extracting CONCEPT-level activations...")
        print("-" * 40)

        # Extract for both train and test splits
        for split in ["train", "test"]:
            stimuli_path = stimuli_dir / f"concept_stimuli_{split}.json"
            if not stimuli_path.exists():
                stimuli_path = stimuli_dir / "concept_stimuli.json"

            scenarios = PrivacyStimulusGenerator.load(str(stimuli_path))
            print(f"\n  [{split}] {len(scenarios)} scenarios")

            result = extractor.extract_concept_activations(scenarios)
            extractor.save_activations(result, f"concept_{split}")

            # Print layer activation shapes
            sample_layer = list(result["activations"].keys())[0]
            print(f"  Activation shape per layer: {result['activations'][sample_layer].shape}")

    # --- Function-level activations (RQ2) ---
    if not args.skip_function:
        print("\n" + "-" * 40)
        print("Extracting FUNCTION-level activations...")
        print("-" * 40)

        func_path = stimuli_dir / "function_stimuli.json"
        function_stimuli = PrivacyStimulusGenerator.load(str(func_path))
        print(f"  {len(function_stimuli)} stimulus pairs")

        result = extractor.extract_function_activations(function_stimuli)
        extractor.save_activations(result, "function")

        # Print stats
        sample_layer = list(result["activations"].keys())[0]
        print(f"  Activation shape per layer: {result['activations'][sample_layer].shape}")

    # --- CI decomposition activations (RQ4) ---
    if not args.skip_ci:
        print("\n" + "-" * 40)
        print("Extracting CI DECOMPOSITION activations...")
        print("-" * 40)

        ci_path = stimuli_dir / "ci_decomposition_stimuli.json"
        ci_stimuli = PrivacyStimulusGenerator.load(str(ci_path))
        print(f"  Parameters: {list(ci_stimuli.keys())}")

        result = extractor.extract_ci_decomposition_activations(ci_stimuli)

        for param_name, data in result.items():
            extractor.save_activations(data, f"ci_{param_name}")
            sample_layer = list(data["activations"].keys())[0]
            print(f"  {param_name}: {data['activations'][sample_layer].shape}")

    print("\n" + "=" * 60)
    print("Activation extraction complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    # Clean up GPU memory
    del model, helper
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
