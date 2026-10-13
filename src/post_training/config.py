"""Serializable settings; no model or GPU initialization at import time."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SFTJobConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    model_revision: str = "main"
    num_train_epochs: int = 5
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    max_length: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    seed: int = 42
    save_steps: int = 4
    device: str = "cuda"
    precision: str = "bf16"

    def __post_init__(self):
        for name in ("num_train_epochs", "batch_size", "gradient_accumulation_steps",
                     "max_length", "lora_rank", "lora_alpha", "save_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("Use fp32 for the CPU smoke run")

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))
