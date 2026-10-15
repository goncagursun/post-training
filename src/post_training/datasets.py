"""Resolve administrator-registered demo or local Bitext datasets."""
from post_training.training.sft import MINI_QA

DEMO_DATASETS = {
    "demo://tenant-a/mini-qa-v1": MINI_QA[:2],
    "demo://tenant-b/mini-qa-v1": MINI_QA[2:],
}


def load_demo_dataset(uri: str, tenant_id: str) -> list[tuple[str, str]]:
    if uri != f"demo://{tenant_id}/mini-qa-v1" or uri not in DEMO_DATASETS:
        raise ValueError("Dataset reference is unsupported or does not belong to this tenant")
    return list(DEMO_DATASETS[uri])


def load_registered_dataset(uri: str, tenant_id: str):
    if uri.startswith("demo://"):
        return load_demo_dataset(uri, tenant_id), None, {"uri": uri}
    from urllib.parse import unquote, urlparse
    from post_training.bitext import intent_pairs, load_manifest
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc:
        raise ValueError("Only local file manifests and demo datasets are supported")
    manifest, splits = load_manifest(unquote(parsed.path), tenant_id)
    return intent_pairs(splits["train"]), intent_pairs(splits["validation"]), manifest
