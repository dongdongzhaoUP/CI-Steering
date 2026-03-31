#!/usr/bin/env python3
"""Generate concept-level, function-level, and CI decomposition stimulus datasets."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.stimulus_generation import PrivacyStimulusGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate CI-Steering stimuli")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pairs-per-type", type=int, default=50,
                        help="Number of concept-level pairs per information type")
    parser.add_argument("--num-function-pairs", type=int, default=200,
                        help="Number of function-level stimulus pairs")
    parser.add_argument("--num-ci-per-condition", type=int, default=100,
                        help="Number of CI decomposition stimuli per condition")
    parser.add_argument("--output-dir", type=str, default="data/stimuli")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = PrivacyStimulusGenerator(seed=args.seed)

    # 1. Concept-level stimuli
    print("=" * 60)
    print("Generating concept-level stimuli...")
    print("=" * 60)
    concept_stimuli = generator.generate_concept_stimuli(
        num_pairs_per_type=args.num_pairs_per_type,
    )
    generator.save(concept_stimuli, output_dir / "concept_stimuli.json")

    num_appropriate = sum(1 for s in concept_stimuli if s["is_appropriate"])
    num_inappropriate = sum(1 for s in concept_stimuli if not s["is_appropriate"])
    print(f"  Total: {len(concept_stimuli)} scenarios")
    print(f"  Appropriate: {num_appropriate}, Inappropriate: {num_inappropriate}")

    from collections import Counter
    type_counts = Counter(s["info_type"] for s in concept_stimuli)
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # 2. Function-level stimuli
    print("\n" + "=" * 60)
    print("Generating function-level stimuli...")
    print("=" * 60)
    function_stimuli = generator.generate_function_stimuli(
        num_pairs=args.num_function_pairs,
    )
    generator.save(function_stimuli, output_dir / "function_stimuli.json")
    print(f"  Total: {len(function_stimuli)} paired stimuli")

    # Show a sample
    s0 = function_stimuli[0]
    print(f"\n  Sample scenario (person={s0['person']}, confidant={s0['confidant']}):")
    print(f"    System: {s0['system_msg'][:100]}...")
    print(f"    User: {s0['user_msg'][:150]}...")
    print(f"    Secret keywords: {s0['secret_keywords']}")

    # 3. CI decomposition stimuli
    print("\n" + "=" * 60)
    print("Generating CI decomposition stimuli...")
    print("=" * 60)
    ci_stimuli = generator.generate_ci_decomposition_stimuli(
        num_per_condition=args.num_ci_per_condition,
    )
    generator.save(ci_stimuli, output_dir / "ci_decomposition_stimuli.json")

    for param, stimuli in ci_stimuli.items():
        unique_values = set(s["varied_value"] for s in stimuli)
        print(f"  {param}: {len(stimuli)} stimuli, {len(unique_values)} unique values")

    # 4. Train/test split metadata (stratified by info_type)
    print("\n" + "=" * 60)
    print("Creating stratified train/test split...")
    print("=" * 60)

    import random
    rng = random.Random(args.seed)

    pair_to_info = {}
    for s in concept_stimuli:
        pair_to_info[s["pair_id"]] = s["info_type"]

    from collections import defaultdict
    info_to_pairs: dict[str, list[int]] = defaultdict(list)
    for pid, itype in pair_to_info.items():
        info_to_pairs[itype].append(pid)

    train_pair_ids: set[int] = set()
    test_pair_ids: set[int] = set()
    for itype, pids in info_to_pairs.items():
        rng.shuffle(pids)
        split = int(len(pids) * 0.8)
        train_pair_ids.update(pids[:split])
        test_pair_ids.update(pids[split:])

    train_concept = [s for s in concept_stimuli if s["pair_id"] in train_pair_ids]
    test_concept = [s for s in concept_stimuli if s["pair_id"] in test_pair_ids]

    generator.save(train_concept, output_dir / "concept_stimuli_train.json")
    generator.save(test_concept, output_dir / "concept_stimuli_test.json")
    print(f"  Train: {len(train_concept)} scenarios ({len(train_pair_ids)} pairs)")
    print(f"  Test:  {len(test_concept)} scenarios ({len(test_pair_ids)} pairs)")

    train_types = Counter(s["info_type"] for s in train_concept)
    test_types = Counter(s["info_type"] for s in test_concept)
    print(f"  Train info_types: {dict(sorted(train_types.items()))}")
    print(f"  Test  info_types: {dict(sorted(test_types.items()))}")

    print("\n" + "=" * 60)
    print(f"All stimuli saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
