"""Pinned, offline CPU embedding adapter using attention-mask mean pooling."""

import hashlib
import importlib
from pathlib import Path
from typing import Any

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]
MAX_TOKENS = 128


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def snapshot(cache: Path, offline: bool) -> Path:
    hub = importlib.import_module("huggingface_hub")
    return Path(
        hub.snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=str(cache),
            allow_patterns=FILES,
            local_files_only=offline,
        )
    )


class Encoder:
    def __init__(self, cache: Path, threads: int = 2) -> None:
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError("CPU threads must be between 1 and 4")
        path = snapshot(cache, offline=True)
        self.files = {name: digest(path / name) for name in FILES}
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(threads)
        self.torch.manual_seed(1729)
        transformers = importlib.import_module("transformers")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.model = transformers.AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        self.model.to("cpu")
        self.model.eval()

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[int]]:
        vectors: list[list[float]] = []
        lengths: list[int] = []
        for start in range(0, len(texts), 8):
            batch = texts[start : start + 8]
            untruncated: Any = self.tokenizer(batch, truncation=False)
            lengths.extend(len(ids) for ids in untruncated["input_ids"])
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
            )
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vectors.extend(pooled.tolist())
        return vectors, lengths
