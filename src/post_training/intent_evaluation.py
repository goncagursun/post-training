"""Greedy generated-label accuracy, using identical data and decoding for comparisons."""
import json
from pathlib import Path

from post_training.bitext import INTENTS, intent_question, load_manifest
from post_training.training.sft import format_prompt


def normalize_label(text):
    return text.strip().lower()


def score_predictions(rows, predictions):
    if not rows or len(rows) != len(predictions):
        raise ValueError("Expected one prediction per nonempty example")
    correct = sum(normalize_label(p) == row["intent"] for row, p in zip(rows, predictions))
    per_intent = {}
    for label in INTENTS:
        pairs = [(row, p) for row, p in zip(rows, predictions) if row["intent"] == label]
        per_intent[label] = {"count": len(pairs), "accuracy": (
            sum(normalize_label(p) == row["intent"] for row, p in pairs) / len(pairs)
            if pairs else None)}
    return {"count": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "invalid_label_rate": sum(normalize_label(p) not in INTENTS for p in predictions) / len(rows),
            "per_intent": per_intent}


def evaluate_intents(manifest_path, split="validation", adapter_dir=None, device="cpu", limit=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftConfig, PeftModel

    if split not in {"validation", "test"}:
        raise ValueError("Evaluate validation or test only")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Selected device is unavailable")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    manifest, splits = load_manifest(manifest_path)
    rows = splits[split] if limit is None else splits[split][:limit]
    model_name, revision = "HuggingFaceTB/SmolLM2-135M", "main"
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir).expanduser().resolve()
        saved_manifest = json.loads((adapter_dir.parent / "dataset.json").read_text())
        if saved_manifest != manifest:
            raise ValueError("Adapter dataset lineage does not match the evaluation manifest")
        config = json.loads((adapter_dir.parent / "config.json").read_text())
        model_name, revision = config["model_name"], config["model_revision"]
        peft_config = PeftConfig.from_pretrained(str(adapter_dir), local_files_only=True)
        if peft_config.base_model_name_or_path != model_name:
            raise ValueError("Base-model metadata mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir) if adapter_dir else model_name,
        **({"local_files_only": True} if adapter_dir else {"revision": revision}))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.float32)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False, local_files_only=True)
    model.to(device).eval()
    model.config.use_cache = True

    def predict():
        predictions = []
        for row in rows:
            inputs = tokenizer(format_prompt(intent_question(row["instruction"])),
                               add_special_tokens=False, return_tensors="pt").to(device)
            n = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                    num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            predictions.append(tokenizer.decode(output[0, n:], skip_special_tokens=True).strip())
        return predictions

    if adapter_dir:
        with model.disable_adapter():
            base_predictions = predict()
        adapter_predictions = predict()
    else:
        base_predictions = predict()
        adapter_predictions = None
    report = {"manifest": manifest, "split": split, "partial": limit is not None,
              "model_name": model_name, "model_revision": revision, "device": device,
              "adapter_dir": str(adapter_dir) if adapter_dir else None,
              "decoding": {"do_sample": False, "max_new_tokens": 16},
              "scoring": "exact label after stripping whitespace and lowercasing",
              "base": score_predictions(rows, base_predictions)}
    if adapter_predictions is not None:
        report["adapter"] = score_predictions(rows, adapter_predictions)
    report["predictions"] = [
        {**row, "base_prediction": base_predictions[i],
         **({"adapter_prediction": adapter_predictions[i]} if adapter_predictions is not None else {})}
        for i, row in enumerate(rows)]
    return report
