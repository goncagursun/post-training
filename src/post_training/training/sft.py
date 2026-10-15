"""One PEFT LoRA training run, independent of Ray and the job queue."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from post_training.config import SFTJobConfig


# Original four examples: a mechanics smoke dataset, not a quality benchmark.
MINI_QA = [
    ("What does a KV cache store?", "A KV cache stores previously computed key and value tensors during autoregressive generation."),
    ("What is gradient accumulation?", "Gradient accumulation combines gradients from multiple micro-batches before performing an optimizer update."),
    ("What does dropout do?", "Dropout randomly disables activations during training as a regularization technique."),
    ("What is LoRA?", "LoRA fine-tunes a model by learning low-rank updates while keeping the original model weights frozen."),
]


def format_prompt(question: str) -> str:
    """Use this same prefix for later generation/evaluation."""
    return f"Question: {question}\nAnswer:"


@dataclass(frozen=True)
class TrainingResult:
    adapter_dir: str
    global_step: int
    training_loss: float


def run_sft(config: SFTJobConfig, run_dir: str | Path,
            examples: list[tuple[str, str]] | None = None,
            validation_examples: list[tuple[str, str]] | None = None) -> TrainingResult:
    # Lazy imports let config/CLI inspection work without the GPU dependencies.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this config. Run on RunPod or select cpu/fp32 explicitly.")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; select fp32 explicitly.")

    run_dir = Path(run_dir).expanduser().resolve()
    # Each invocation owns a fresh directory. Resume will be a separate increment.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision)
    if tokenizer.eos_token is None:
        raise ValueError("This training format requires an EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = MINI_QA if examples is None else examples
    if not examples:
        raise ValueError("Training dataset is empty")
    def make_dataset(pairs):
        return Dataset.from_list([
            {"prompt": format_prompt(question), "completion": " " + answer + tokenizer.eos_token}
            for question, answer in pairs
        ])

    dataset = make_dataset(examples)
    validation_dataset = make_dataset(validation_examples) if validation_examples else None
    # Refuse silent truncation, including evaluation data.
    for row in list(dataset) + (list(validation_dataset) if validation_dataset is not None else []):
        ids = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"]
        if len(ids) > config.max_length:
            raise ValueError("max_length would truncate a training example")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        torch_dtype=torch.bfloat16 if config.precision == "bf16" else torch.float32,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        max_grad_norm=1.0,
        max_length=config.max_length,
        completion_only_loss=True,
        packing=False,
        bf16=config.precision == "bf16",
        fp16=False,
        use_cpu=config.device == "cpu",
        dataloader_pin_memory=config.device == "cuda",
        seed=config.seed,
        data_seed=config.seed,
        logging_steps=1,
        save_strategy="steps",
        save_steps=config.save_steps if len(examples) <= 4 else max(config.save_steps, 100),
        save_total_limit=2,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                         eval_dataset=validation_dataset, processing_class=tokenizer)
    output = trainer.train()
    if validation_dataset is not None:
        validation_metrics = trainer.evaluate()
        (run_dir / "validation.json").write_text(json.dumps(validation_metrics, indent=2) + "\n")
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()
    result = TrainingResult(str(adapter_dir), trainer.state.global_step, float(output.training_loss))
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(output.metrics, indent=2) + "\n")
    return result
