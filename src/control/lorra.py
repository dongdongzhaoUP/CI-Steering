"""
Low-Rank Representation Adaptation (LoRRA) for privacy steering.

Trains a LoRA adapter so that its hidden states match:
    target = orig_hidden + α * (pos_hidden - neg_hidden)
at target layers, then merges the adapter into the base model.
"""

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def _build_lorra_templates(tokenizer):
    """
    Return ``(orig_fn, pos_fn, neg_fn)`` callables that format
    ``(instruction, response, control=None)`` into chat strings.

    Uses ``format_chat_training`` from model_utils so the format
    automatically matches the loaded model.
    """
    from ..utils.model_utils import format_chat_training

    def orig_fn(instruction: str, response: str, **_kw) -> str:
        return format_chat_training(
            tokenizer,
            system_msg="You are a helpful assistant.",
            user_msg=instruction,
            response=response,
        )

    def pos_fn(instruction: str, response: str, control: str = "", **_kw) -> str:
        user_msg = f"{instruction} {control}".strip()
        return format_chat_training(
            tokenizer,
            system_msg="You are a helpful assistant.",
            user_msg=user_msg,
            response=response,
        )

    def neg_fn(instruction: str, response: str, control: str = "", **_kw) -> str:
        user_msg = f"{instruction} {control}".strip()
        return format_chat_training(
            tokenizer,
            system_msg="You are a helpful assistant.",
            user_msg=user_msg,
            response=response,
        )

    return orig_fn, pos_fn, neg_fn


@dataclass
class LoRRAConfig:
    """Configuration for LoRRA training."""

    # Model
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Privacy concept
    pos_type: str = "a privacy-preserving"
    neg_type: str = "a privacy-violating"
    control_template: str = "Give {type} response when discussing personal information."

    # Target layers for representation matching
    target_layers: list[int] = field(default_factory=lambda: [10, 12, 14, 16, 18, 20])

    # LoRRA hyperparameters
    lorra_alpha: float = 5.0  # Scaling factor for direction shift
    max_res_len: int = 64     # Max response length for hidden state extraction

    # LoRA hyperparameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Training
    num_train_examples: int = 5000
    max_steps: int = 100
    learning_rate: float = 3e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    bf16: bool = True
    model_max_length: int = 256
    output_dir: str = "outputs/lorra"

    def to_dict(self) -> dict:
        """Serialize config to dict."""
        return {k: v for k, v in self.__dict__.items()}


class LoRRAPrivacyDataset(Dataset):
    """Creates (orig, pos, neg) triplets for LoRRA training."""

    def __init__(
        self,
        tokenizer,
        config: LoRRAConfig,
        instructions: Optional[list[str]] = None,
        responses: Optional[list[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.max_res_len = config.max_res_len

        # Build control strings
        self.pos_control = config.control_template.format(type=config.pos_type)
        self.neg_control = config.control_template.format(type=config.neg_type)

        # Load data
        if instructions is not None and responses is not None:
            self.instructions = instructions[: config.num_train_examples]
            self.responses = responses[: config.num_train_examples]
        else:
            self.instructions, self.responses = self._load_alpaca(
                config.num_train_examples
            )

        # Build model-agnostic template functions
        orig_fn, pos_fn, neg_fn = _build_lorra_templates(tokenizer)

        # Precompute triplets
        self.orig_texts = []
        self.pos_texts = []
        self.neg_texts = []

        for instr, resp in zip(self.instructions, self.responses):
            # Truncate response to max_res_len tokens
            resp_tokens = tokenizer.encode(resp, add_special_tokens=False)
            resp_truncated = tokenizer.decode(
                resp_tokens[: self.max_res_len], skip_special_tokens=True
            )

            self.orig_texts.append(
                orig_fn(instruction=instr, response=resp_truncated)
            )
            self.pos_texts.append(
                pos_fn(
                    instruction=instr,
                    control=self.pos_control,
                    response=resp_truncated,
                )
            )
            self.neg_texts.append(
                neg_fn(
                    instruction=instr,
                    control=self.neg_control,
                    response=resp_truncated,
                )
            )

        print(f"  LoRRA dataset: {len(self)} samples")
        print(f"  Pos control: '{self.pos_control}'")
        print(f"  Neg control: '{self.neg_control}'")

    @staticmethod
    def _load_alpaca(num_examples: int) -> tuple[list[str], list[str]]:
        """Load instruction-response pairs from Alpaca dataset."""
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split="train")
        # Filter to instruction-only (no additional input)
        ds = ds.filter(lambda x: x["input"] == "")

        instructions = ds["instruction"][:num_examples]
        outputs = ds["output"][:num_examples]
        return instructions, outputs

    def __len__(self):
        return len(self.orig_texts)

    def __getitem__(self, idx):
        orig = self.orig_texts[idx]
        pos = self.pos_texts[idx]
        neg = self.neg_texts[idx]

        tokenized = self.tokenizer(
            [orig, pos, neg],
            padding="max_length",
            truncation=True,
            max_length=self.config.model_max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],       # (3, seq_len)
            "attention_mask": tokenized["attention_mask"],  # (3, seq_len)
        }


def lorra_compute_loss(
    trainer_self,
    model,
    inputs,
    target_layers: list[int],
    alpha: float,
    max_res_len: int = 64,
    return_outputs: bool = False,
    **kwargs,
):
    """LoRRA loss: L2(lora_hidden, orig_hidden + α * (pos_hidden - neg_hidden))."""
    input_ids = inputs["input_ids"]       # (batch, 3, seq_len)
    attention_mask = inputs["attention_mask"]  # (batch, 3, seq_len)

    if input_ids.shape[1] != 3:
        raise ValueError(f"Expected triplet input, got shape {input_ids.shape}")

    orig_input_ids = input_ids[:, 0]      # (batch, seq_len)
    pos_input_ids = input_ids[:, 1]
    neg_input_ids = input_ids[:, 2]

    orig_attn_mask = attention_mask[:, 0]
    pos_attn_mask = attention_mask[:, 1]
    neg_attn_mask = attention_mask[:, 2]

    min_length = max_res_len
    response_mask = orig_attn_mask[:, -min_length:]
    response_mask = (
        response_mask
        .repeat(len(target_layers), 1, 1)
        .unsqueeze(-1)
    )

    # hidden_states[l+1] = output of model.layers[l]
    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            orig_outputs = model(
                input_ids=orig_input_ids,
                attention_mask=orig_attn_mask,
                output_hidden_states=True,
            )["hidden_states"]
            orig_hidden = [
                orig_outputs[l + 1][:, -min_length:].detach()
                for l in target_layers
            ]

            pos_outputs = model(
                input_ids=pos_input_ids,
                attention_mask=pos_attn_mask,
                output_hidden_states=True,
            )["hidden_states"]

            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attn_mask,
                output_hidden_states=True,
            )["hidden_states"]

            direction_hidden = [
                pos_outputs[l + 1][:, -min_length:].detach()
                - neg_outputs[l + 1][:, -min_length:].detach()
                for l in target_layers
            ]

            target_hidden = torch.stack([
                orig_hidden[i] + alpha * direction_hidden[i]
                for i in range(len(target_layers))
            ]) * response_mask  # (n_layers, batch, min_length, hidden)

    del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    lora_outputs = model(
        input_ids=orig_input_ids,
        attention_mask=orig_attn_mask,
        output_hidden_states=True,
    )["hidden_states"]

    lora_hidden = torch.stack([
        lora_outputs[l + 1][:, -min_length:]
        for l in target_layers
    ]) * response_mask  # (n_layers, batch, min_length, hidden)

    loss = torch.norm(
        lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float
    ).nanmean()

    return (loss, lora_hidden) if return_outputs else loss


class LoRRATrainer:
    """Manages LoRRA training: setup, train, and merge/save the adapter."""

    def __init__(self, config: LoRRAConfig):
        if LoraConfig is None:
            raise ImportError(
                "peft package not installed. Run: pip install peft"
            )

        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def setup(self):
        """Load model, tokenizer, and prepare LoRA."""
        print(f"Loading model: {self.config.model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            padding_side="left",
            model_max_length=self.config.model_max_length,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        compute_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=compute_dtype,
            device_map="auto",
        )

        lora_layers_to_transform = list(
            range(max(self.config.target_layers) + 1)
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

        print(f"  Target layers for representation matching: {self.config.target_layers}")
        print(f"  LoRA applied to layers: 0..{max(self.config.target_layers)}")
        print(f"  LoRRA alpha: {self.config.lorra_alpha}")

    def train(
        self,
        instructions: Optional[list[str]] = None,
        responses: Optional[list[str]] = None,
    ):
        """Run LoRRA training (uses Alpaca data if instructions/responses are None)."""
        if self.peft_model is None:
            self.setup()

        dataset = LoRRAPrivacyDataset(
            tokenizer=self.tokenizer,
            config=self.config,
            instructions=instructions,
            responses=responses,
        )

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="constant",
            bf16=self.config.bf16,
            logging_steps=10,
            save_total_limit=0,
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,
            weight_decay=0.0,
        )

        target_layers = self.config.target_layers
        lorra_alpha = self.config.lorra_alpha
        max_res_len = self.config.max_res_len

        class LoRRACustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return lorra_compute_loss(
                    self,
                    model,
                    inputs,
                    target_layers=target_layers,
                    alpha=lorra_alpha,
                    max_res_len=max_res_len,
                    return_outputs=return_outputs,
                )

        trainer = LoRRACustomTrainer(
            model=self.peft_model,
            processing_class=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
        )

        self.peft_model.config.use_cache = False
        print("\nStarting LoRRA training...")
        trainer.train()
        print("LoRRA training complete!")

    def save_merged_model(self, output_dir: str) -> Path:
        """Merge LoRA into base model and save."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Merging LoRA adapter and saving to {output_path}...")
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(output_path / "lorra_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"  Saved merged model to {output_path}")
        return output_path

    def save_adapter(self, output_dir: str):
        """Save just the LoRA adapter (without merging)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.peft_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(output_path / "lorra_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        print(f"  Saved LoRA adapter to {output_path}")
        return output_path
