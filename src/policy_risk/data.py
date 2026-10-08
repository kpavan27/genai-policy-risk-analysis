"""ToxicChat loading."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://huggingface.co/datasets/lmsys/toxic-chat/resolve/refs%2Fconvert%2Fparquet/"
    "toxicchat0124/train/0000.parquet"
)
SHA256 = "7700b88519ded5fef3062a1175bb2b1df6f3c459b4557aa4f7d05c7051f35d5a"

MODERATION_CATEGORIES = [
    "sexual",
    "sexual/minors",
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "violence",
    "violence/graphic",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
]


def fetch(raw_dir: str | Path) -> Path:
    path = Path(raw_dir) / "toxicchat0124_train.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as resp:
            path.write_bytes(resp.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"checksum mismatch: {digest}")
    return path


def parse_moderation(raw: str) -> dict[str, float]:
    return {cat: float(score) for cat, score in json.loads(raw)}


def load(raw_dir: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(fetch(raw_dir))
    scores = pd.DataFrame([parse_moderation(s) for s in df["openai_moderation"]])
    missing = set(MODERATION_CATEGORIES) - set(scores.columns)
    if missing:
        raise ValueError(f"moderation categories missing: {missing}")
    scores = scores[MODERATION_CATEGORIES].add_prefix("mod:")
    out = pd.concat([df.drop(columns=["openai_moderation", "model_output"]), scores], axis=1)
    out["mod_max"] = scores.max(axis=1)
    out["mod_top"] = scores.idxmax(axis=1).str.removeprefix("mod:")
    return out


def human_annotated(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["human_annotation"]].reset_index(drop=True)
