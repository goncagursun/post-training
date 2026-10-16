"""One PEFT LoRA training run, independent of Ray and the job queue."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from post_training.config import SFTJobConfig
from post_training.recovery import (TrainingInterrupted, fingerprint, publish_checkpoint,
                                    validate_checkpoint)


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
    training_loss: float | None
    telemetry: dict = field(default_factory=dict)


def _run_sft(config: SFTJobConfig, run_dir: str | Path,
            examples: list[tuple[str, str]] | None = None,
            validation_examples: list[tuple[str, str]] | None = None, telemetry=None,
            resume_checkpoint=None, interrupt_after_checkpoint=None) -> TrainingResult:
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
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    set_seed(config.seed)

    with telemetry.phase("tokenizer_loading"):
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

    with telemetry.phase("dataset_preparation"):
        dataset = make_dataset(examples)
        validation_dataset = make_dataset(validation_examples) if validation_examples else None
        # Refuse silent truncation, including evaluation data.
        for row in list(dataset) + (list(validation_dataset) if validation_dataset is not None else []):
            ids = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"]
            if len(ids) > config.max_length:
                raise ValueError("max_length would truncate a training example")

    with telemetry.phase("model_loading_and_adapters"):
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


    run_fingerprint = fingerprint(config, examples, validation_examples,
                                  getattr(model.config, "_commit_hash", None))
    initial_step = 0
    if resume_checkpoint:
        initial_step = validate_checkpoint(resume_checkpoint, run_fingerprint)["global_step"]
        telemetry.values["resume"] = {"checkpoint": str(resume_checkpoint), "global_step": initial_step}
    synchronize = torch.cuda.synchronize if config.device == "cuda" else None

    class InstrumentedTrainer(SFTTrainer):
        input_tokens = 0
        supervised_tokens = 0

        def training_step(self, model, inputs, *args, **kwargs):
            self.samples_observed = getattr(self, "samples_observed", 0) + inputs["input_ids"].shape[0]
            self.input_tokens += inputs["attention_mask"].sum().detach()
            self.supervised_tokens += (inputs["labels"][:, 1:] != -100).sum().detach()
            return super().training_step(model, inputs, *args, **kwargs)

        def _save_checkpoint(self, *args, **kwargs):
            with telemetry.phase("checkpoint_writing", synchronize):
                result = super()._save_checkpoint(*args, **kwargs)
                checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                publish_checkpoint(checkpoint, run_fingerprint)
                telemetry.event("checkpoint_completed", step=self.state.global_step, path=str(checkpoint))
            if interrupt_after_checkpoint and self.state.global_step >= interrupt_after_checkpoint:
                raise TrainingInterrupted("Stopped after a completed checkpoint for the recovery demo")
            return result

    from transformers import TrainerCallback

    class LogMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            telemetry.event("trainer_metrics", step=state.global_step, metrics=logs or {})

    with telemetry.phase("trainer_setup"):
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
            skip_memory_metrics=True,
        )
        trainer = InstrumentedTrainer(model=model, args=args, train_dataset=dataset,
                             eval_dataset=validation_dataset, processing_class=tokenizer, callbacks=[LogMetrics()])
    with telemetry.phase("training", synchronize):
        target_steps = trainer.args.num_train_epochs * math.ceil(
            math.ceil(len(dataset) / config.batch_size) / config.gradient_accumulation_steps)
        if resume_checkpoint and initial_step >= target_steps:
            # A crash after the final checkpoint still needs validation/export, not
            # another optimizer update. No new training loss exists for this attempt.
            from transformers.trainer_callback import TrainerState
            from transformers.trainer_utils import TrainOutput
            trainer._load_from_checkpoint(str(resume_checkpoint))
            trainer.state = TrainerState.load_from_json(str(Path(resume_checkpoint) / "trainer_state.json"))
            output = TrainOutput(initial_step, 0.0, {})
        else:
            output = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    training_seconds = telemetry.phases["training"]
    input_tokens = int(trainer.input_tokens)
    supervised_tokens = int(trainer.supervised_tokens)
    observed_steps = trainer.state.global_step - initial_step
    telemetry.values["training"] = {
        "observed_optimizer_steps": observed_steps,
        "optimizer_steps": trainer.state.global_step,
        "input_tokens": input_tokens, "supervised_tokens": supervised_tokens,
        "samples": getattr(trainer, "samples_observed", 0),
        "samples_per_second": getattr(trainer, "samples_observed", 0) / training_seconds,
        "input_tokens_per_second": input_tokens / training_seconds,
        "supervised_tokens_per_second": supervised_tokens / training_seconds,
        "optimizer_steps_per_second": observed_steps / training_seconds,
        "mean_seconds_per_optimizer_step": training_seconds / observed_steps if observed_steps else None,
        "throughput_scope": "trainer.train wall time including checkpointing",
    }
    if validation_dataset is not None:
        with telemetry.phase("validation", synchronize):
            validation_metrics = trainer.evaluate()
            (run_dir / "validation.json").write_text(json.dumps(validation_metrics, indent=2) + "\n")
    adapter_dir = run_dir / "adapter"
    with telemetry.phase("adapter_export", synchronize):
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        trainer.save_state()
    # Trainer divides accumulated loss by global_step even when resuming.
    attempt_loss = (float(output.training_loss) * trainer.state.global_step / observed_steps
                    if observed_steps else None)
    output.metrics["train_loss"] = attempt_loss
    telemetry.values["training"]["loss_scope"] = "updates performed in this attempt only"
    result = TrainingResult(str(adapter_dir), trainer.state.global_step, attempt_loss)
    (run_dir / "metrics.json").write_text(json.dumps(output.metrics, indent=2) + "\n")
    return result


def run_sft(config: SFTJobConfig, run_dir: str | Path,
            examples: list[tuple[str, str]] | None = None,
            validation_examples: list[tuple[str, str]] | None = None,
            context: dict | None = None, resume_checkpoint: str | None = None,
            interrupt_after_checkpoint: int | None = None) -> TrainingResult:
    from post_training.telemetry import RunTelemetry, NvidiaSampler
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    telemetry = RunTelemetry(run_dir, context)
    telemetry.event("run_started", device=config.device)
    sampler = None
    torch = None
    status = "FAILED"
    error_type = None
    try:
        with telemetry.phase("dependency_imports"):
            import torch
        telemetry.values["device"] = config.device
        if config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            telemetry.values["gpu_name"] = props.name
            sampler = NvidiaSampler(telemetry, getattr(props, "uuid", None))
            sampler.start()
        result = _run_sft(config, run_dir, examples, validation_examples, telemetry,
                          resume_checkpoint, interrupt_after_checkpoint)
        status = "SUCCEEDED"
    except BaseException as exc:
        error_type = type(exc).__name__
        if isinstance(exc, TrainingInterrupted):
            status = "INTERRUPTED"
        raise
    finally:
        telemetry.values["gpu_activity"] = sampler.stop() if sampler else {
            "status": "not_applicable" if config.device == "cpu" else "unavailable", "sample_count": 0}
        if torch is not None and config.device == "cuda" and torch.cuda.is_available():
            telemetry.values["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            telemetry.values["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        else:
            telemetry.values["cuda_peak_allocated_bytes"] = None
            telemetry.values["cuda_peak_reserved_bytes"] = None
        summary = telemetry.finish(status, error_type)
    result = TrainingResult(result.adapter_dir, result.global_step, result.training_loss, summary)
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result
