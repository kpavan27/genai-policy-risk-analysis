"""Runs the analysis and writes reports/."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from . import analysis, data  # noqa: E402
from .stats import wilson  # noqa: E402

log = logging.getLogger(__name__)


def run(config_path: str, raw_dir: str = "data/raw", out_dir: str = "reports") -> dict[str, Any]:
    policy = yaml.safe_load(Path(config_path).read_text())
    thr = policy["flag_threshold"]
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    df = data.load(raw_dir)
    h = data.human_annotated(df)
    metrics: dict[str, Any] = {
        "dataset": {
            "prompts": int(len(df)),
            "human_annotated": int(len(h)),
            "non_annotated_all_labelled_non_toxic": bool(
                (df.loc[~df.human_annotation, "toxicity"] == 0).all()
            ),
            "non_annotated_max_mod_score_p99": float(df.loc[~df.human_annotation, "mod_max"].quantile(0.99)),
        }
    }

    cov = analysis.coverage_table(policy)
    cov.to_csv(out / "tables" / "policy_coverage.csv", index=False)
    metrics["coverage"] = cov["moderation_coverage"].value_counts().to_dict()

    metrics["moderation_quality"] = mq = analysis.moderation_quality(h, thr)
    for key in ("toxicity_at_policy_threshold", "toxicity_at_best_f1_threshold"):
        t = mq[key]["threshold"]
        mq[key]["flagged_per_1k_prompts"] = float(1000 * (h["mod_max"] >= t).mean())
    missed_toxic = h[(h["toxicity"] == 1) & (h["mod_max"] < thr)]
    mq["missed_toxic_top_category"] = missed_toxic["mod_top"].value_counts().head(5).to_dict()

    det = analysis.compile_detectors(policy)
    hits_all = analysis.detector_hits(df["user_input"], det)
    gaps = analysis.gap_sizing(df, hits_all, thr)
    ranking = analysis.risk_ranking(gaps, policy)
    ranking.to_csv(out / "tables" / "risk_ranking.csv", index=False)
    metrics["risk_ranking"] = ranking.drop(columns=["moderation_flag_ci"]).to_dict(orient="records")

    hits_h = analysis.detector_hits(h["user_input"], det)
    hits_h_v1 = analysis.detector_hits(h["user_input"], analysis.compile_detectors(policy, "v1"))
    metrics["jailbreak_detector"] = {
        "v1": analysis.jailbreak_detector_validation(h, hits_h_v1["2c"]),
        "current": analysis.jailbreak_detector_validation(h, hits_h["2c"]),
    }

    reg = h[hits_h["1e"]]
    metrics["regulated_goods_human_annotated"] = {
        "n": int(len(reg)),
        "human_toxic": int(reg["toxicity"].sum()),
        "moderation_flagged": int((reg["mod_max"] >= thr).sum()),
        "max_mod_score": float(reg["mod_max"].max()) if len(reg) else float("nan"),
        "flag_rate_ci": list(wilson(int((reg["mod_max"] >= thr).sum()), int(len(reg)))),
    }

    missed = h[(h["toxicity"] == 1) & (h["mod_max"] < thr)]
    metrics["missed_toxic"] = {
        "n": int(len(missed)),
        "share_jailbreak": float(missed["jailbreaking"].mean()),
        "themes_top_terms": analysis.miss_themes(missed["user_input"]),
    }

    _fig_score_ecdf(h, thr, out / "figures" / "moderation_scores_by_label.png")
    _fig_coverage(cov, out / "figures" / "policy_coverage.png")
    _fig_risk(ranking, out / "figures" / "risk_ranking.png")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("wrote %s", out / "metrics.json")
    return metrics


def _fig_score_ecdf(h: pd.DataFrame, thr: float, path: Path) -> None:
    groups = {
        "not toxic": h[h.toxicity == 0],
        "toxic (not jailbreak)": h[(h.toxicity == 1) & (h.jailbreaking == 0)],
        "jailbreak": h[h.jailbreaking == 1],
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, g in groups.items():
        s = np.sort(np.clip(g["mod_max"].to_numpy(), 1e-6, 1))
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", label=f"{label} (n={len(g)})")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"flag threshold {thr}")
    ax.set_xscale("log")
    ax.set(
        xlabel="Highest moderation category score (log scale)",
        ylabel="Share of prompts at or below score",
        title="Most human-labelled toxic prompts score far below the flag threshold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_coverage(cov: pd.DataFrame, path: Path) -> None:
    colours = {"covered": "#4c9a6a", "partial": "#e0b040", "none": "#c8553d"}
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y = np.arange(len(cov))[::-1]
    ax.barh(y, 1, color=[colours[s] for s in cov["moderation_coverage"]])
    for yi, (_, r) in zip(y, cov.iterrows(), strict=True):
        ax.text(0.02, yi, f"{r['clause']}  {r['title']}", va="center", fontsize=8, color="white")
    ax.set(yticks=[], xticks=[], xlim=(0, 1), title="Policy clauses with a moderation signal")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(
        handles,
        ["covered", "partially covered", "no signal"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_risk(r: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    labels = (
        r["clauses"].str.replace(r", e\.g\. personal data without consent", "", regex=True).str.slice(0, 70)
    )
    ax.barh(labels[::-1], r["unflagged_per_1k"][::-1], color="#c8553d")
    for i, (_, row) in enumerate(r[::-1].iterrows()):
        ax.text(
            row["unflagged_per_1k"] + 0.1,
            i,
            f"{row['hits']} hits, {row['moderation_flag_rate']:.0%} flagged",
            va="center",
            fontsize=7,
        )
    ax.set(
        xlabel="Prompts per 1,000 matching the clause and NOT flagged by moderation",
        title="Uncovered policy areas, sized on real traffic",
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
