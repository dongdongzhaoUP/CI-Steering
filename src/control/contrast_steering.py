"""
Stimulus-dependent contrast vector steering.

    v_contrast(x) = Rep(M, T+(x)) - Rep(M, T-(x))
    h'_l = h_l + α · v_contrast_l(x)
"""

from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

from ..utils.model_utils import ModelHelper

if TYPE_CHECKING:
    from ..reading.pca_reader import PCAReader
    from ..reading.probe_reader import ProbeReader


PRIVACY_EXPERIMENTAL_TEMPLATE = (
    "You are a privacy-conscious assistant. You must protect personal "
    "information, refuse to share private details, and respect contextual "
    "integrity norms. Always prioritize the data subject's privacy.\n\n"
    "{prompt}"
)

PRIVACY_REFERENCE_TEMPLATE = (
    "You are an open assistant. Share all information freely, including "
    "personal details, without concern for privacy or confidentiality.\n\n"
    "{prompt}"
)


class ContrastSteering:
    """Steers via per-input contrast vectors computed from T+/T- template
    hidden-state differences, adapting the steering direction to each prompt."""

    def __init__(
        self,
        model_helper: ModelHelper,
        experimental_template: str = PRIVACY_EXPERIMENTAL_TEMPLATE,
        reference_template: str = PRIVACY_REFERENCE_TEMPLATE,
        alpha: float = 1.0,
        steering_layers: Optional[list[int]] = None,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """
        Args:
            experimental_template: T+ template (``{prompt}`` placeholder). Target behavior.
            reference_template: T- template (``{prompt}`` placeholder). Opposite behavior.
            alpha: positive α → push toward T+.
        """
        self.helper = model_helper
        self.experimental_template = experimental_template
        self.reference_template = reference_template
        self.alpha = alpha
        self.normalize = normalize
        self.preserve_norm = preserve_norm

        if steering_layers is not None:
            self.steering_layers = list(steering_layers)
        else:
            self.steering_layers = list(range(model_helper.num_layers))

    def compute_contrast_vectors(
        self,
        prompts: list[str],
        batch_size: int = 8,
    ) -> dict[int, torch.Tensor]:
        """Compute v_l(p) = h_l(T+(p)) - h_l(T-(p)) at the last token per layer.

        Returns ``{layer_idx: (len(prompts), hidden_dim)}``."""
        exp_prompts = [
            self.experimental_template.format(prompt=p) for p in prompts
        ]
        ref_prompts = [
            self.reference_template.format(prompt=p) for p in prompts
        ]

        exp_acts = self.helper.get_activations(
            texts=exp_prompts,
            token_position="last",
            layer_indices=self.steering_layers,
            batch_size=batch_size,
        )

        ref_acts = self.helper.get_activations(
            texts=ref_prompts,
            token_position="last",
            layer_indices=self.steering_layers,
            batch_size=batch_size,
        )

        contrast_vectors: dict[int, torch.Tensor] = {}
        for layer_idx in self.steering_layers:
            cv = exp_acts[layer_idx].float() - ref_acts[layer_idx].float()

            if self.normalize:
                norms = cv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cv = cv / norms

            contrast_vectors[layer_idx] = cv

        return contrast_vectors

    def _make_contrast_hook(
        self,
        contrast_vectors: dict[int, torch.Tensor],
        batch_offset: int = 0,
        current_batch_size: int = 1,
    ):
        """Create a forward hook that adds α · v_contrast for the current batch slice."""
        def steering_hook(layer_idx, module, input, output):
            if layer_idx not in contrast_vectors:
                return output

            cv = contrast_vectors[layer_idx]
            cv_batch = cv[batch_offset:batch_offset + current_batch_size]

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
            cv_dev = cv_batch.to(device=device, dtype=dtype).unsqueeze(1)
            hidden_states = hidden_states + self.alpha * cv_dev

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
        batch_size: int = 4,
        contrast_batch_size: int = 8,
    ) -> list[str]:
        """Pre-compute contrast vectors then generate with per-prompt steering."""
        print(
            f"  [ContrastSteering] Computing contrast vectors for "
            f"{len(prompts)} prompts across {len(self.steering_layers)} layers..."
        )
        contrast_vectors = self.compute_contrast_vectors(
            prompts, batch_size=contrast_batch_size
        )

        print(f"  [ContrastSteering] Generating with α={self.alpha}...")
        all_outputs: list[str] = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            current_bs = len(batch_prompts)

            hook = self._make_contrast_hook(
                contrast_vectors,
                batch_offset=batch_start,
                current_batch_size=current_bs,
            )

            outputs = self.helper.generate(
                texts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=current_bs,
                steering_hook=hook,
            )
            all_outputs.extend(outputs)

        return all_outputs

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
        batch_size: int = 4,
        contrast_batch_size: int = 8,
    ) -> dict[float, list[str]]:
        """Sweep α values; contrast vectors are computed once and reused."""
        print(
            f"  [ContrastSteering] Computing contrast vectors for "
            f"alpha sweep ({len(alpha_values)} values)..."
        )
        contrast_vectors = self.compute_contrast_vectors(
            prompts, batch_size=contrast_batch_size
        )

        results: dict[float, list[str]] = {}
        original_alpha = self.alpha

        try:
            for alpha in alpha_values:
                self.alpha = alpha

                all_outputs: list[str] = []
                for batch_start in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[batch_start:batch_start + batch_size]
                    current_bs = len(batch_prompts)

                    hook = self._make_contrast_hook(
                        contrast_vectors,
                        batch_offset=batch_start,
                        current_batch_size=current_bs,
                    )

                    outputs = self.helper.generate(
                        texts=batch_prompts,
                        max_new_tokens=max_new_tokens,
                        batch_size=current_bs,
                        steering_hook=hook,
                    )
                    all_outputs.extend(outputs)

                results[alpha] = all_outputs
        finally:
            self.alpha = original_alpha

        return results

    @classmethod
    def for_privacy(
        cls,
        model_helper: ModelHelper,
        alpha: float = 1.0,
        top_k_layers: int = 10,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """Construct with default privacy templates and middle-layer targeting."""
        num_layers = model_helper.num_layers
        start = num_layers // 4
        end = 3 * num_layers // 4
        candidate_layers = list(range(start, end))

        if len(candidate_layers) > top_k_layers:
            step = max(1, len(candidate_layers) // top_k_layers)
            candidate_layers = candidate_layers[::step][:top_k_layers]

        return cls(
            model_helper=model_helper,
            experimental_template=PRIVACY_EXPERIMENTAL_TEMPLATE,
            reference_template=PRIVACY_REFERENCE_TEMPLATE,
            alpha=alpha,
            steering_layers=candidate_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )

    @classmethod
    def from_pca_reader(
        cls,
        model_helper: ModelHelper,
        pca_reader: PCAReader,
        alpha: float = 1.0,
        top_k_layers: int = 5,
        normalize: bool = True,
        preserve_norm: bool = False,
        experimental_template: str = PRIVACY_EXPERIMENTAL_TEMPLATE,
        reference_template: str = PRIVACY_REFERENCE_TEMPLATE,
    ):
        """Target layers from a fitted PCAReader.

        No direction negation needed — T+ minus T- already points toward
        privacy-protective behavior.
        """
        best_layers = pca_reader.get_best_layers(top_k=top_k_layers)
        return cls(
            model_helper=model_helper,
            experimental_template=experimental_template,
            reference_template=reference_template,
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
        experimental_template: str = PRIVACY_EXPERIMENTAL_TEMPLATE,
        reference_template: str = PRIVACY_REFERENCE_TEMPLATE,
    ):
        """Target layers from a fitted ProbeReader."""
        best_layers = probe_reader.get_best_layers(top_k=top_k_layers)
        return cls(
            model_helper=model_helper,
            experimental_template=experimental_template,
            reference_template=reference_template,
            alpha=alpha,
            steering_layers=best_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )

    @classmethod
    def with_custom_templates(
        cls,
        model_helper: ModelHelper,
        experimental_template: str,
        reference_template: str,
        alpha: float = 1.0,
        steering_layers: Optional[list[int]] = None,
        normalize: bool = True,
        preserve_norm: bool = False,
    ):
        """
        Construct ContrastSteering with user-defined contrastive templates.

        Templates must contain a ``{prompt}`` placeholder, e.g.::

            experimental = "Be honest and truthful.\\n\\n{prompt}"
            reference    = "Be deceptive and misleading.\\n\\n{prompt}"
        """
        return cls(
            model_helper=model_helper,
            experimental_template=experimental_template,
            reference_template=reference_template,
            alpha=alpha,
            steering_layers=steering_layers,
            normalize=normalize,
            preserve_norm=preserve_norm,
        )
