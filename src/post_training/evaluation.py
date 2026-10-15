"""Artifact reload smoke check; held-out quality evaluation comes later."""
from dataclasses import dataclass
import json
from pathlib import Path

from post_training.training.sft import format_prompt


@dataclass(frozen=True)
class GenerationResult:
    adapter_dir: str
    base_model: str
    base_revision: str
    device: str
    prompt: str
    answer: str
    generated_tokens: int


def generate_adapter(adapter_dir: str | Path, prompt: str, device: str = "cpu",
                     max_new_tokens: int = 64) -> GenerationResult:
    """Reload a local exported adapter and generate greedily in inference mode."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    adapter_dir = Path(adapter_dir).expanduser().resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu on your Mac")

    adapter_config = PeftConfig.from_pretrained(str(adapter_dir), local_files_only=True)
    base_model = adapter_config.base_model_name_or_path
    if not base_model:
        raise ValueError("Adapter configuration does not identify its base model")
    revision = adapter_config.revision or "main"
    # Our export stores the runtime config next to adapter/, not inside it.
    config_path = adapter_dir.parent / "config.json"
    if config_path.is_file():
        training_config = json.loads(config_path.read_text())
        if training_config["model_name"] != base_model:
            raise ValueError("Saved run config and adapter disagree about the base model")
        revision = training_config.get("model_revision", revision)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, revision=revision, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False,
                                      local_files_only=True)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    inputs = tokenizer(format_prompt(prompt), add_special_tokens=False, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and prompt_length + max_new_tokens > context_limit:
        raise ValueError("Prompt plus generation budget exceeds the model context window")
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, prompt_length:]
    return GenerationResult(
        adapter_dir=str(adapter_dir), base_model=base_model, base_revision=revision,
        device=device, prompt=prompt,
        answer=tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        generated_tokens=len(new_tokens),
    )
