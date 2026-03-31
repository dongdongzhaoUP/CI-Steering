"""Activation extractor: wraps stimuli in LAT templates and extracts hidden states via ModelHelper."""

import json
import torch
from pathlib import Path
from typing import Optional

from ..data.stimulus_generation import PrivacyStimulusGenerator
from ..utils.model_utils import ModelHelper


class ActivationExtractor:
    """
    Extracts and stores hidden-state activations for privacy stimuli.
    Core infrastructure for the LAT pipeline adapted for contextual privacy.
    """

    def __init__(
        self,
        model_helper: ModelHelper,
        output_dir: str = "outputs/activations",
        token_position: str = "last",
        batch_size: int = 8,
    ):
        self.helper = model_helper
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token_position = token_position
        self.batch_size = batch_size

    def extract_concept_activations(
        self,
        scenarios: list[dict],
        layer_indices: Optional[list[int]] = None,
    ) -> dict:
        """Extract activations for concept-level stimuli."""
        # Wrap each scenario in the LAT concept template
        texts = [
            PrivacyStimulusGenerator.wrap_concept_template(s["text"])
            for s in scenarios
        ]

        labels = [s["is_appropriate"] for s in scenarios]
        pair_ids = [s["pair_id"] for s in scenarios]
        info_types = [s["info_type"] for s in scenarios]

        print(f"Extracting concept activations for {len(texts)} scenarios...")
        activations = self.helper.get_activations(
            texts=texts,
            token_position=self.token_position,
            layer_indices=layer_indices,
            batch_size=self.batch_size,
        )

        return {
            "activations": activations,
            "labels": labels,
            "pair_ids": pair_ids,
            "info_types": info_types,
            "metadata": scenarios,
        }

    def extract_function_activations(
        self,
        function_stimuli: list[dict],
        layer_indices: Optional[list[int]] = None,
    ) -> dict:
        """Extract activations for function-level (social roleplay) stimuli."""
        texts = [
            PrivacyStimulusGenerator.wrap_function_template(s)
            for s in function_stimuli
        ]

        print(f"Extracting function activations for {len(texts)} stimuli...")
        activations = self.helper.get_activations(
            texts=texts,
            token_position=self.token_position,
            layer_indices=layer_indices,
            batch_size=self.batch_size,
        )

        return {
            "activations": activations,
            "metadata": function_stimuli,
        }

    def extract_ci_decomposition_activations(
        self,
        ci_stimuli: dict[str, list[dict]],
        layer_indices: Optional[list[int]] = None,
    ) -> dict[str, dict]:
        """Extract activations for CI parameter decomposition stimuli."""
        results = {}

        for param_name, stimuli in ci_stimuli.items():
            texts = [
                PrivacyStimulusGenerator.wrap_concept_template(s["text"])
                for s in stimuli
            ]
            varied_values = [s["varied_value"] for s in stimuli]

            print(f"Extracting CI decomposition activations for '{param_name}' "
                  f"({len(texts)} stimuli)...")
            activations = self.helper.get_activations(
                texts=texts,
                token_position=self.token_position,
                layer_indices=layer_indices,
                batch_size=self.batch_size,
            )

            results[param_name] = {
                "activations": activations,
                "varied_values": varied_values,
                "metadata": stimuli,
            }

        return results

    def save_activations(self, data: dict, name: str):
        """
        Save extracted activations to disk.

        Activations (tensors) are saved as .pt files.
        Metadata is saved as .json.
        """
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save activations
        act_keys = [k for k in data if "activations" in k.lower() and isinstance(data[k], dict)]
        for key in act_keys:
            act_dict = data[key]
            act_save = {str(layer): tensor for layer, tensor in act_dict.items()}
            torch.save(act_save, save_dir / f"{key}.pt")

        # Save non-tensor data as JSON
        meta = {k: v for k, v in data.items() if k not in act_keys}
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(f"Saved activations to {save_dir}")

    @staticmethod
    def load_activations(path: str) -> dict:
        """Load activations and metadata from disk."""
        load_dir = Path(path)

        result = {}

        # Load .pt files
        for pt_file in load_dir.glob("*.pt"):
            key = pt_file.stem
            act_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
            result[key] = {int(layer): tensor for layer, tensor in act_dict.items()}

        # Load metadata
        meta_file = load_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            result.update(meta)

        return result
