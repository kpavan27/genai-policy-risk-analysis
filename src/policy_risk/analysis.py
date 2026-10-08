"""Coverage, detection quality, gap sizing and risk ranking."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_recall_curve

from .stats import wilson


def coverage_table(policy: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for cid, meta in policy["clauses"].items():
        full = [c for c, m in policy["moderation_crosswalk"].items() if m.get(cid) == "full"]
        partial = [c for c, m in policy["moderation_crosswalk"].items() if m.get(cid) == "partial"]
        status = "covered" if full else "partial" if partial else "none"
        rows.append(
            {
                "clause": cid,
                "section": meta["section"],
                "title": meta["title"],
                "moderation_coverage": status,
                "moderation_categories": ", ".join(full + partial),
            }
        )
    return pd.DataFrame(rows)


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    lo, hi = wilson(tp, tp + fn)
    return {
        "threshold": threshold,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "recall_ci": [lo, hi],
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def moderation_quality(h: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Moderation max score against human labels."""
    tox = h["toxicity"].to_numpy() == 1
    jb = h["jailbreaking"].to_numpy() == 1
    s = h["mod_max"].to_numpy()
    t_best = best_f1_threshold(tox, s)
    out = {
        "n_human_annotated": int(len(h)),
        "n_toxic": int(tox.sum()),
        "n_jailbreak": int(jb.sum()),
        "toxicity_pr_auc": float(average_precision_score(tox, s)),
        "toxicity_at_policy_threshold": threshold_metrics(tox, s, threshold),
        "toxicity_at_best_f1_threshold": threshold_metrics(tox, s, t_best),
        "jailbreak_recall_at_policy_threshold": threshold_metrics(jb, s, threshold),
        "jailbreak_recall_at_best_f1_threshold": threshold_metrics(jb, s, t_best),
        "jailbreak_median_mod_max": float(np.median(s[jb])),
        "toxic_non_jailbreak_median_mod_max": float(np.median(s[tox & ~jb])),
    }
    caught = h[tox & (s >= threshold)]
    out["top_category_of_flagged_toxic"] = caught["mod_top"].value_counts().to_dict()
    return out


def compile_detectors(policy: dict[str, Any], version: str = "current") -> dict[str, re.Pattern]:
    """``version="v1"`` re-adds patterns removed in revisions."""
    detectors = {k: list(v) for k, v in policy["detectors"].items()}
    if version == "v1":
        for key, rev in policy.get("detector_revisions", {}).items():
            detectors[key] = detectors[key] + rev["removed"]
    return {
        key: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE) for key, pats in detectors.items()
    }


def detector_hits(texts: pd.Series, detectors: dict[str, re.Pattern]) -> pd.DataFrame:
    return pd.DataFrame(
        {k: texts.map(lambda t, rx=rx: rx.search(t) is not None) for k, rx in detectors.items()},
        index=texts.index,
    )


def gap_sizing(df: pd.DataFrame, hits: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    n_all = len(df)
    for key in hits.columns:
        m = hits[key]
        sub = df[m]
        human = sub[sub["human_annotation"]]
        n = int(m.sum())
        flagged = int((sub["mod_max"] >= threshold).sum())
        lo, hi = wilson(flagged, n)
        rows.append(
            {
                "detector": key,
                "hits": n,
                "per_1k_prompts": 1000 * n / n_all,
                "moderation_flag_rate": flagged / n if n else float("nan"),
                "moderation_flag_ci": [lo, hi],
                "human_annotated_hits": int(len(human)),
                "human_toxic_rate_among_hits": float(human["toxicity"].mean())
                if len(human)
                else float("nan"),
                "human_jailbreak_rate_among_hits": float(human["jailbreaking"].mean())
                if len(human)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("hits", ascending=False).reset_index(drop=True)


def jailbreak_detector_validation(h: pd.DataFrame, hits_h: pd.Series) -> dict[str, float]:
    y = h["jailbreaking"].to_numpy() == 1
    pred = hits_h.to_numpy()
    tp, fp, fn = int(np.sum(pred & y)), int(np.sum(pred & ~y)), int(np.sum(~pred & y))
    return {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "moderation_recall_same_prompts": float(np.mean(h.loc[y, "mod_max"] >= 0.5)),
        "union_recall": float(np.mean((pred | (h["mod_max"].to_numpy() >= 0.5))[y])),
    }


def risk_ranking(gaps: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    sev = {
        key: max(policy["clauses"][c]["severity"] for c in clauses)
        for key, clauses in policy["detector_clauses"].items()
    }
    names = {
        key: " / ".join(f"{c} {policy['clauses'][c]['title']}" for c in clauses)
        for key, clauses in policy["detector_clauses"].items()
    }
    r = gaps.copy()
    r["clauses"] = r["detector"].map(names)
    r["severity"] = r["detector"].map(sev)
    r["unflagged_per_1k"] = (r["per_1k_prompts"] * (1 - r["moderation_flag_rate"])).fillna(0.0)
    r["risk_score"] = r["unflagged_per_1k"] * r["severity"]
    return r.sort_values("risk_score", ascending=False).reset_index(drop=True)


def miss_themes(texts: pd.Series, n_topics: int = 6, top_terms: int = 8, seed: int = 0) -> list[list[str]]:
    """NMF topic terms."""
    vec = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.6, ngram_range=(1, 2))
    x = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=500)
    nmf.fit(x)
    terms = vec.get_feature_names_out()
    return [[terms[i] for i in comp.argsort()[::-1][:top_terms]] for comp in nmf.components_]
