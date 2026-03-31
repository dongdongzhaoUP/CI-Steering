"""Model loading, chat formatting, and hook-based activation extraction for CI-Steering."""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union


def resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolve OpenAI API key from explicit arg, env var, or .api_key file."""
    if explicit_key:
        return explicit_key
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    key_file = Path(".api_key")
    if key_file.exists():
        return key_file.read_text().strip() or None
    return None


def format_chat_prompt(
    tokenizer,
    system_msg: str,
    user_msg: str,
    assistant_prefill: str = "",
) -> str:
    """Build a chat prompt using the tokenizer's chat template, with plain-text fallback."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    # If there is an assistant prefill, add it as an assistant message
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})

    has_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    if has_template:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not assistant_prefill,
            )
            return prompt
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            pass  # fall through to manual fallback

    # Fallback for base models (no chat template)
    parts = []
    if system_msg:
        parts.append(f"System: {system_msg}\n\n")
    parts.append(f"User: {user_msg}\n\n")
    if assistant_prefill:
        parts.append(f"Assistant: {assistant_prefill}")
    else:
        parts.append("Assistant:")
    return "".join(parts)


def format_chat_training(
    tokenizer,
    system_msg: str,
    user_msg: str,
    response: str,
) -> str:
    """Build a complete chat example (with response) for training datasets."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "content": response})

    has_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    if has_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            pass

    # Fallback
    parts = []
    if system_msg:
        parts.append(f"System: {system_msg}\n\n")
    parts.append(f"User: {user_msg}\n\n")
    parts.append(f"Assistant: {response}")
    return "".join(parts)


def default_target_layers(num_layers: int, k: int = 6) -> list[int]:
    """Pick *k* evenly-spaced layers from the middle-to-late region (strongest concept representations)."""
    start = num_layers // 4
    end = 3 * num_layers // 4
    candidates = list(range(start, end))
    if len(candidates) <= k:
        return candidates
    step = max(1, len(candidates) // k)
    return candidates[::step][:k]


def load_model(
    model_name: str,
    dtype: str = "float16",
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
):
    """Load a HuggingFace causal LM and its tokenizer."""
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",   # important for batch generation with decoder models
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    model.eval()

    return model, tokenizer


class ModelHelper:
    """Hook-based hidden-state extraction from transformer models."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._hooks = []
        self._activations = {}

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        if hasattr(self.model, "model"):
            # LlamaForCausalLM, MistralForCausalLM, etc.
            if hasattr(self.model.model, "layers"):
                return len(self.model.model.layers)
        if hasattr(self.model, "transformer"):
            # GPT-J, GPT-2, etc.
            if hasattr(self.model.transformer, "h"):
                return len(self.model.transformer.h)
        raise ValueError("Cannot determine number of layers for this model architecture.")

    @property
    def hidden_size(self) -> int:
        """Return the hidden dimension of the model."""
        return self.model.config.hidden_size

    def _get_layers(self):
        """Return the list of transformer layer modules."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        raise ValueError("Cannot access transformer layers for this model architecture.")

    def register_hooks(self, layer_indices: Optional[list[int]] = None):
        """Register forward hooks on specified layers to capture hidden states."""
        self.clear_hooks()
        layers = self._get_layers()

        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        for idx in layer_indices:
            layer = layers[idx]

            def hook_fn(module, input, output, layer_idx=idx):
                # output is a tuple; first element is the hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self._activations[layer_idx] = hidden_states.detach()

            handle = layer.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._activations = {}

    def get_activations(
        self,
        texts: list[str],
        token_position: Union[str, int] = "last",
        layer_indices: Optional[list[int]] = None,
        batch_size: int = 8,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden-state activations for a batch of texts."""
        self.register_hooks(layer_indices)

        all_layer_acts = {}
        num_layers = self.num_layers
        target_layers = layer_indices or list(range(num_layers))

        for layer_idx in target_layers:
            all_layer_acts[layer_idx] = []

        try:
            for batch_start in range(0, len(texts), batch_size):
                batch_texts = texts[batch_start:batch_start + batch_size]
                self._activations = {}

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    self.model(**inputs)

                # Extract activations at specified token position
                attention_mask = inputs["attention_mask"]

                for layer_idx in target_layers:
                    if layer_idx not in self._activations:
                        continue

                    hidden = self._activations[layer_idx]  # (batch, seq_len, hidden)

                    if token_position == "last":
                        # With left-padding, real tokens are right-aligned,
                        # so the last real token is always at position -1.
                        if self.tokenizer.padding_side == "left":
                            extracted = hidden[:, -1, :]  # (batch, hidden)
                        else:
                            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
                            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                            extracted = hidden[batch_indices, seq_lengths]

                    elif token_position == "concept":
                        # Find the position of "privacy" token in each sequence
                        extracted = self._extract_at_concept_token(
                            hidden, batch_texts, inputs["input_ids"]
                        )

                    elif isinstance(token_position, int):
                        extracted = hidden[:, token_position, :]

                    else:
                        # Default to last token
                        if self.tokenizer.padding_side == "left":
                            extracted = hidden[:, -1, :]
                        else:
                            seq_lengths = attention_mask.sum(dim=1) - 1
                            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                            extracted = hidden[batch_indices, seq_lengths]

                    all_layer_acts[layer_idx].append(extracted.cpu())

            for layer_idx in target_layers:
                if all_layer_acts[layer_idx]:
                    all_layer_acts[layer_idx] = torch.cat(all_layer_acts[layer_idx], dim=0)
                else:
                    all_layer_acts[layer_idx] = torch.empty(0, self.hidden_size)

        finally:
            self.clear_hooks()

        return all_layer_acts

    def _extract_at_concept_token(
        self,
        hidden: torch.Tensor,
        texts: list[str],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract activations at the 'privacy' token position."""
        # Encode with and without leading space to handle BPE tokenizers
        # that produce different IDs for "privacy" vs " privacy"
        target_ids = set()
        for variant in ["privacy", " privacy"]:
            target_ids.update(
                self.tokenizer.encode(variant, add_special_tokens=False)
            )

        batch_size = hidden.size(0)
        extracted = []

        for i in range(batch_size):
            ids = input_ids[i].tolist()
            positions = [p for p, t in enumerate(ids) if t in target_ids]
            if positions:
                pos = positions[-1]  # Use last occurrence
            else:
                # Fallback to last real token (use attention_mask for correctness)
                pos = hidden.size(1) - 1
            extracted.append(hidden[i, pos, :])

        return torch.stack(extracted, dim=0)

    def generate(
        self,
        texts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 8,
        steering_hook: Optional[callable] = None,
    ) -> list[str]:
        """Generate completions, optionally with a steering hook applied at each layer."""
        all_outputs = []
        handles = []

        if steering_hook is not None:
            layers = self._get_layers()
            for idx, layer in enumerate(layers):
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        return steering_hook(layer_idx, module, input, output)
                    return hook_fn
                handle = layer.register_forward_hook(make_hook(idx))
                handles.append(handle)

        try:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for batch_idx, batch_start in enumerate(range(0, len(texts), batch_size)):
                batch_texts = texts[batch_start:batch_start + batch_size]
                print(f"  [generate] batch {batch_idx+1}/{total_batches} ({batch_start+1}-{min(batch_start+len(batch_texts), len(texts))}/{len(texts)})", flush=True)

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                else:
                    gen_kwargs["do_sample"] = False

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        **gen_kwargs,
                    )

                input_len = inputs["input_ids"].shape[1]
                for i in range(len(batch_texts)):
                    new_tokens = output_ids[i, input_len:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    all_outputs.append(text.strip())

        finally:
            for handle in handles:
                handle.remove()

        return all_outputs
