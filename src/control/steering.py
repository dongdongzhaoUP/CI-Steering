"""
Privacy steering via representation control: h'_l = h_l + α * v_privacy_l.

Adds privacy direction vectors to residual-stream hidden states during inference.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Union, Callable, TYPE_CHECKING

from ..utils.model_utils import ModelHelper

if TYPE_CHECKING:
    from ..reading.pca_reader import PCAReader
    from ..reading.probe_reader import ProbeReader


class PrivacySteering:
    """
    Steers model behavior toward privacy-preserving outputs by adding
    privacy direction vectors to hidden states during inference.
    """

    def __init__(
        self,
        model_helper: ModelHelper,
        privacy_directions: dict[int, Union[np.ndarray, torch.Tensor]],
        alpha: float = 1.0,
        steering_layers: Optional[list[int]] = None,
        normalize: bool = True,
        preserve_norm: bool = False,
        layer_weights: Optional[dict[int, float]] = None,
    ):
        """
        Args:
            layer_weights: per-layer weights from probe accuracy (normalized
                so max=1.0). Concentrates steering on layers with strongest
                privacy encoding. None = uniform weighting.
        """
        self.helper = model_helper
        self.alpha = alpha
        self.normalize = normalize
        self.preserve_norm = preserve_norm

        self.layer_weights = layer_weights or {}
        self.privacy_directions = {}
        for layer_idx, direction in privacy_directions.items():
            if isinstance(direction, np.ndarray):
                direction = torch.from_numpy(direction)
            direction = direction.float()
            if normalize:
                direction = direction / direction.norm()
            self.privacy_directions[layer_idx] = direction

        if steering_layers is not None:
            self.steering_layers = set(steering_layers)
        else:
            self.steering_layers = set(self.privacy_directions.keys())

    def _get_layer_weight(self, layer_idx: int) -> float:
        """Probe-derived weight normalized so max is 1.0 (default 1.0)."""
        if not self.layer_weights:
            return 1.0
        max_w = max(self.layer_weights.values())
        w = self.layer_weights.get(layer_idx, 0.0)
        return w / max_w if max_w > 0 else 1.0

    def _make_steering_hook(self) -> Callable:
        """Return hook: h'_l = h_l + α * w_l * v_l."""
        def steering_hook(layer_idx, module, input, output):
            if layer_idx not in self.steering_layers:
                return output

            direction = self.privacy_directions.get(layer_idx)
            if direction is None:
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
            steering_vec = direction.to(device=device, dtype=dtype)

            w_l = self._get_layer_weight(layer_idx)
            steering_vec = steering_vec.unsqueeze(0).unsqueeze(0)
            hidden_states = hidden_states + self.alpha * w_l * steering_vec

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
        """Generate with privacy steering applied."""
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

    def sweep_alpha(
        self,
        prompts: list[str],
        alpha_values: list[float],
        max_new_tokens: int = 256,
        batch_size: int = 8,
    ) -> dict[float, list[str]]:
        """Generate across a range of α values for Pareto-frontier exploration."""
        results = {}
        original_alpha = self.alpha

        try:
            for alpha in alpha_values:
                self.alpha = alpha
                outputs = self.generate(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                )
                results[alpha] = outputs
        finally:
            self.alpha = original_alpha

        return results

    @classmethod
    def from_pca_reader(
        cls,
        model_helper: ModelHelper,
        pca_reader: PCAReader,
        alpha: float = 1.0,
        top_k_layers: int = 5,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """Construct from a fitted PCAReader.

        PCA direction points toward "sharing is OK" — negated here so
        positive α → more privacy-protective.
        """
        best_layers = pca_reader.get_best_layers(top_k=top_k_layers)
        directions = {
            layer: -pca_reader.get_privacy_vector(layer)
            for layer in best_layers
        }
        return cls(
            model_helper=model_helper,
            privacy_directions=directions,
            alpha=alpha,
            steering_layers=best_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )

    @classmethod
    def from_probe_reader(
        cls,
        model_helper: ModelHelper,
        probe_reader: ProbeReader,
        alpha: float = 1.0,
        top_k_layers: int = 5,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """Construct from a fitted ProbeReader (direction negated for privacy)."""
        best_layers = probe_reader.get_best_layers(top_k=top_k_layers)
        directions = {
            layer: -probe_reader.get_privacy_direction(layer)
            for layer in best_layers
        }
        return cls(
            model_helper=model_helper,
            privacy_directions=directions,
            alpha=alpha,
            steering_layers=best_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )

    @classmethod
    def from_probe_reader_weighted(
        cls,
        model_helper: ModelHelper,
        probe_reader: ProbeReader,
        alpha: float = 1.0,
        top_k_layers: int = 5,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """Like from_probe_reader but weights layers by probe CV accuracy."""
        best_layers = probe_reader.get_best_layers(top_k=top_k_layers)
        directions = {
            layer: -probe_reader.get_privacy_direction(layer)
            for layer in best_layers
        }
        layer_weights = {
            layer: probe_reader.layer_scores[layer]["cv_mean"]
            for layer in best_layers
            if layer in probe_reader.layer_scores
        }
        return cls(
            model_helper=model_helper,
            privacy_directions=directions,
            alpha=alpha,
            steering_layers=best_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
            layer_weights=layer_weights,
        )

    @classmethod
    def from_function_activations(
        cls,
        model_helper: ModelHelper,
        diff_activations: dict[int, torch.Tensor],
        alpha: float = 1.0,
        top_k_layers: Optional[int] = None,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """Construct from mean function-level activation differences (T_f+ - T_f-)."""
        directions = {}
        for layer_idx, diffs in diff_activations.items():
            mean_diff = diffs.float().mean(dim=0).numpy()
            directions[layer_idx] = mean_diff

        steering_layers = None
        if top_k_layers is not None:
            magnitudes = {
                layer: np.linalg.norm(d) for layer, d in directions.items()
            }
            sorted_layers = sorted(magnitudes, key=magnitudes.get, reverse=True)
            steering_layers = sorted_layers[:top_k_layers]

        return cls(
            model_helper=model_helper,
            privacy_directions=directions,
            alpha=alpha,
            steering_layers=steering_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )
