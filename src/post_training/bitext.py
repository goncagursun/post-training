"""Reproducible Bitext intent datasets and verified local manifests."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

SOURCE = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
INTENTS = ("cancel_order", "track_order", "get_refund", "change_shipping_address", "recover_password")


def normalize_request(text):
    text = re.sub(r"\{\{.*?\}\}", " placeholder ", text.lower())
    return " ".join(re.findall(r"[a-z]+", text))


def intent_question(instruction, labels=INTENTS):
    return ("Classify the customer request. Reply with exactly one intent label. "
            f"Allowed labels: {', '.join(labels)}.\nRequest: {instruction}")


def build_splits(rows, train_size=500, validation_size=100, test_size=100, seed=42):
    sizes = {"train": train_size, "validation": validation_size, "test": test_size}
    if any(n <= 0 or n % len(INTENTS) for n in sizes.values()):
        raise ValueError("Split sizes must be positive multiples of five")
    # Deduplicate globally, rejecting contradictory labels on the same normalized input.
    unique = {}
    for row in rows:
        if row["intent"] not in INTENTS:
            continue
        key = normalize_request(row["instruction"])
        if not key:
            continue
        if key in unique and unique[key]["intent"] != row["intent"]:
            raise ValueError("Conflicting intent labels on duplicate requests")
        unique.setdefault(key, {"instruction": row["instruction"], "intent": row["intent"]})
    buckets = {(tenant, split): [] for tenant in ("tenant-a", "tenant-b") for split in sizes}
    rng = random.Random(seed)
    for label in INTENTS:
        families = defaultdict(list)
        for key, row in sorted(unique.items()):
            if row["intent"] == label:
                # Word-order variants and punctuation/case/placeholder variants stay together.
                family = " ".join(sorted(set(key.split())))
                families[family].append(row)
        groups = list(families.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        remaining = {bucket: sizes[bucket[1]] // len(INTENTS) for bucket in buckets}
        for group in groups:
            eligible = [b for b in buckets if remaining[b] > 0]
            if not eligible:
                break
            rng.shuffle(eligible)
            bucket = max(eligible, key=lambda b: remaining[b])
            chosen = group[:remaining[bucket]]
            buckets[bucket].extend(chosen)
            remaining[bucket] -= len(chosen)
            # Unused family members are discarded, never placed in another split/tenant.
        if any(remaining.values()):
            raise ValueError(f"Not enough independent {label} examples for requested sizes: {remaining}")
    for rows in buckets.values():
        rng.shuffle(rows)
    return buckets


def prepare_bitext(rows, output_root, train_size=500, validation_size=100, test_size=100, seed=42):
    buckets = build_splits(rows, train_size, validation_size, test_size, seed)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    manifests = []
    for tenant in ("tenant-a", "tenant-b"):
        directory = output_root / tenant
        directory.mkdir()
        manifest = {"format": "bitext-intent-v1", "tenant_id": tenant,
                    "source": SOURCE, "source_revision": REVISION,
                    "license": "CDLA-Sharing-1.0", "seed": seed, "labels": list(INTENTS), "splits": {}}
        for split in ("train", "validation", "test"):
            data = buckets[tenant, split]
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data)
            path = directory / f"{split}.jsonl"
            path.write_text(content, encoding="utf-8")
            manifest["splits"][split] = {"file": path.name, "count": len(data),
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(path)
    return manifests


def load_manifest(path, tenant_id=None):
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "bitext-intent-v1" or manifest.get("labels") != list(INTENTS):
        raise ValueError("Unsupported Bitext manifest")
    if tenant_id is not None and manifest["tenant_id"] != tenant_id:
        raise ValueError("Dataset manifest belongs to another tenant")
    splits = {}
    seen = set()
    for split in ("train", "validation", "test"):
        info = manifest["splits"][split]
        file = (path.parent / info["file"]).resolve()
        if not file.is_relative_to(path.parent):
            raise ValueError("Dataset file escapes manifest directory")
        content = file.read_bytes()
        if hashlib.sha256(content).hexdigest() != info["sha256"]:
            raise ValueError(f"Checksum mismatch for {split}")
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        if not rows or len(rows) != info["count"]:
            raise ValueError("Empty split or incorrect row count")
        for row in rows:
            if not isinstance(row.get("instruction"), str) or row.get("intent") not in INTENTS:
                raise ValueError("Invalid intent example")
            key = normalize_request(row["instruction"])
            if not key or key in seen:
                raise ValueError("Duplicate request within or across splits")
            seen.add(key)
        splits[split] = rows
    return manifest, splits


def intent_pairs(rows):
    return [(intent_question(row["instruction"]), row["intent"]) for row in rows]
