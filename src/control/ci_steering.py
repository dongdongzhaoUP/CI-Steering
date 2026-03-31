"""
CI-Decomposed Compositional Steering.
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Callable, TYPE_CHECKING
import warnings

from ..utils.model_utils import ModelHelper

if TYPE_CHECKING:
    from ..reading.probe_reader import ProbeReader


class CICompositionalSteering:
    """Steers model behavior using per-CI-parameter direction vectors
    with optional probe-weighted layer selection."""

    CI_PARAMETERS = ("info_type", "recipient", "transmission_principle")

    def __init__(
        self,
        model_helper: ModelHelper,
        ci_directions: dict[str, dict[int, Union[np.ndarray, torch.Tensor]]],
        alphas: Optional[dict[str, float]] = None,
        layer_weights: Optional[dict[int, float]] = None,
        steering_layers: Optional[list[int]] = None,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """
        Args:
            ci_directions: ``{param_name: {layer_idx: direction_vector}}``.
            alphas: Per-parameter steering strengths (default 1.0 each).
            layer_weights: Per-layer weights from probe accuracy (default uniform).
            steering_layers: Layers to steer (default: intersection across all params).
        """
        self.helper = model_helper
        self.normalize = normalize
        self.preserve_norm = preserve_norm

        self.alphas = alphas or {param: 1.0 for param in ci_directions}
        self.layer_weights = layer_weights or {}


        self.ci_directions: dict[str, dict[int, torch.Tensor]] = {}
        for param_name, layer_dirs in ci_directions.items():
            self.ci_directions[param_name] = {}
            for layer_idx, direction in layer_dirs.items():
                if isinstance(direction, np.ndarray):
                    direction = torch.from_numpy(direction)
                direction = direction.float()
                if normalize:
                    norm = direction.norm()
                    if norm > 0:
                        direction = direction / norm
                self.ci_directions[param_name][layer_idx] = direction

        if steering_layers is not None:
            self.steering_layers = set(steering_layers)
        else:
            all_layer_sets = [
                set(dirs.keys()) for dirs in self.ci_directions.values()
            ]
            if all_layer_sets:
                self.steering_layers = set.intersection(*all_layer_sets)
            else:
                self.steering_layers = set()

    def _get_layer_weight(self, layer_idx: int) -> float:
        """Get the probe-derived weight for a layer (defaults to 1.0)."""
        if not self.layer_weights:
            return 1.0
        max_w = max(self.layer_weights.values())
        w = self.layer_weights.get(layer_idx, 0.0)
        return w / max_w if max_w > 0 else 1.0

    def _make_steering_hook(self) -> Callable:
        """Return hook: h'_l = h_l + w_l * Σ_{param} α_{param} * v_{param}_l."""
        def steering_hook(layer_idx, module, input, output):
            if layer_idx not in self.steering_layers:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            if self.preserve_norm:
                norm_pre = torch.norm(hidden_states, dim=-1, keepdim=True)

            device = hidden_states.device
            dtype = hidden_states.dtype

            w_l = self._get_layer_weight(layer_idx)

            for param_name, layer_dirs in self.ci_directions.items():
                if layer_idx not in layer_dirs:
                    continue
                direction = layer_dirs[layer_idx].to(device=device, dtype=dtype)
                alpha_param = self.alphas.get(param_name, 0.0)

                steering_vec = direction.unsqueeze(0).unsqueeze(0)
                hidden_states = hidden_states + w_l * alpha_param * steering_vec

            if self.preserve_norm:
                norm_post = torch.norm(hidden_states, dim=-1, keepdim=True).clamp(min=1e-8)
                hidden_states = hidden_states / norm_post * norm_pre

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return steering_hook

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> list[str]:
        """Generate with CI-decomposed steering applied."""
        hook = self._make_steering_hook()
        return self.helper.generate(
            texts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            steering_hook=hook,
        )

    def generate_unsteered(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> list[str]:
        """Generate without steering (baseline)."""
        return self.helper.generate(
            texts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            steering_hook=None,
        )

    def sweep_alphas(
        self,
        prompts: list[str],
        alpha_configs: list[dict[str, float]],
        max_new_tokens: int = 256,
        batch_size: int = 8,
    ) -> list[dict]:
        """Generate across multiple per-parameter alpha configurations (for ablation)."""
        results = []
        original_alphas = self.alphas.copy()

        try:
            for config in alpha_configs:
                self.alphas = {param: 0.0 for param in self.ci_directions}
                self.alphas.update(config)

                outputs = self.generate(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                )
                results.append({
                    "alphas": config.copy(),
                    "outputs": outputs,
                })
        finally:
            self.alphas = original_alphas

        return results

    def sweep_global_alpha(
        self,
        prompts: list[str],
        alpha_values: list[float],
        max_new_tokens: int = 256,
        batch_size: int = 8,
    ) -> dict[float, list[str]]:
        """Sweep a global multiplier across all CI parameters simultaneously."""
        results = {}
        base_alphas = self.alphas.copy()

        try:
            for global_alpha in alpha_values:
                self.alphas = {
                    param: global_alpha * base_alphas.get(param, 1.0)
                    for param in self.ci_directions
                }
                outputs = self.generate(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                )
                results[global_alpha] = outputs
        finally:
            self.alphas = base_alphas

        return results

    @classmethod
    def from_ci_directions_dir(
        cls,
        model_helper: ModelHelper,
        ci_dir: str,
        alphas: Optional[dict[str, float]] = None,
        layer_weights: Optional[dict[int, float]] = None,
        top_k_layers: int = 5,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """Load CI-decomposed directions from a ci_decomposition.py output directory."""
        ci_path = Path(ci_dir)
        ci_directions = {}

        for param_name in cls.CI_PARAMETERS:
            fname = ci_path / f"ci_{param_name}_directions.pt"
            if not fname.exists():
                warnings.warn(
                    f"[CISteering] Skipping {param_name} (not found: {fname})"
                )
                continue

            raw_dirs = torch.load(fname, map_location="cpu", weights_only=True)
            layer_dirs = {
                (int(k) if isinstance(k, str) else k): v
                for k, v in raw_dirs.items()
            }
            ci_directions[param_name] = layer_dirs
            print(f"  [CISteering] Loaded {param_name}: "
                  f"{len(layer_dirs)} layers")

        if not ci_directions:
            raise FileNotFoundError(
                f"No CI direction files found in {ci_path}. "
                f"Run ci_decomposition.py with --save-directions first."
            )

        layer_magnitudes: dict[int, float] = {}
        for param_dirs in ci_directions.values():
            for layer_idx, d in param_dirs.items():
                mag = d.float().norm().item()
                layer_magnitudes[layer_idx] = (
                    layer_magnitudes.get(layer_idx, 0.0) + mag
                )

        sorted_layers = sorted(
            layer_magnitudes, key=layer_magnitudes.get, reverse=True
        )
        selected_layers = sorted_layers[:top_k_layers]

        for param_name in ci_directions:
            ci_directions[param_name] = {
                l: ci_directions[param_name][l]
                for l in selected_layers
                if l in ci_directions[param_name]
            }

        return cls(
            model_helper=model_helper,
            ci_directions=ci_directions,
            alphas=alphas,
            layer_weights=layer_weights,
            steering_layers=selected_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )

    @classmethod
    def from_probe_reader_with_weights(
        cls,
        model_helper: ModelHelper,
        ci_directions: dict[str, dict[int, Union[np.ndarray, torch.Tensor]]],
        probe_reader: ProbeReader,
        alphas: Optional[dict[str, float]] = None,
        top_k_layers: int = 5,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """Construct CI steering with probe-accuracy-weighted layers."""
        best_layers = probe_reader.get_best_layers(top_k=top_k_layers)
        layer_weights = {
            layer: probe_reader.layer_scores[layer]["cv_mean"]
            for layer in best_layers
            if layer in probe_reader.layer_scores
        }

        return cls(
            model_helper=model_helper,
            ci_directions=ci_directions,
            alphas=alphas,
            layer_weights=layer_weights,
            steering_layers=best_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )

    def describe(self) -> str:
        """Return a human-readable summary of the steering configuration."""
        lines = ["CI-Decomposed Compositional Steering"]
        lines.append(f"  Parameters: {list(self.ci_directions.keys())}")
        lines.append(f"  Alphas: {self.alphas}")
        lines.append(f"  Steering layers: {sorted(self.steering_layers)}")
        if self.layer_weights:
            lines.append(f"  Layer weights (raw): {self.layer_weights}")
            lines.append(f"  Layer weights (normalized):")
            for l in sorted(self.steering_layers):
                lines.append(f"    Layer {l}: {self._get_layer_weight(l):.4f}")
        else:
            lines.append("  Layer weights: uniform (1.0)")
        lines.append(f"  Preserve norm: {self.preserve_norm}")
        return "\n".join(lines)
