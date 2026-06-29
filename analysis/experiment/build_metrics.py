"""Build the interactive cohort explorer's data file (cohort_data.js).

This is the one piece of the analysis that stays in Python: it labels each
trace's path shape with the sklearn path-shape classifier (snake/spiral/
random_walk) and bundles every user's runs into the JSON the browser explorer
(cohort_explorer.html) reads. The *statistical* analytics + figures now live
entirely in R (analysis.R), computed straight from the raw experiment CSVs, so
there are no per-trace / per-cell intermediate CSVs to keep in sync anymore.

Reads:
  - mow_test_rows.csv    raw traces (result = JSON [{x,y,t}, ...] per play)
  - mow_users_rows.csv   demographics, one row per user
  - data/labelled_paths.csv  training data for the path-shape classifier
  - optimal_paths.csv (via level_optima/build_optimal.py)  Concorde optimum/level

Writes:
  - cohort_data.js       window.COHORT = {...} for the interactive cohort explorer

Optimality is measured against the provably-optimal covering walk for each
level, found with the exact Concorde TSP solver (see build_optimal.py):
    optimality = optimal_moves / player_moves    (1.0 = optimal, ->0 worse)
    redundancy = player_moves / optimal_moves    (1.0 = optimal, >1 worse)
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # analysis/experiment -> repo root
sys.path.insert(0, str(REPO))

from classifier import train_and_evaluate  # noqa: E402

from build_optimal import level_optima  # noqa: E402  (same-folder helper)

TEST_CSV = HERE / "mow_test_rows.csv"
USERS_CSV = HERE / "mow_users_rows.csv"
LABELLED_CSV = REPO / "data" / "labelled_paths.csv"

OUT_COHORT = HERE / "cohort_data.js"

# Logical round ordering (difficulty / grid size grows along this axis).
LEVEL_ORDER = ["tutorial", "round1", "round2", "bonus1", "bonus2", "bonus3"]

DEMO_COLS = ["age", "style", "gaming", "hand", "optimization"]

# A move that took longer than this (ms) to make is a deliberate "thinking"
# move; faster ones are fluent "execution". Kept in sync with PAUSE_MS in
# analysis.R (the 1000ms reference also used in eda.R).
PAUSE_MS = 1000


def points_to_moves(pts: list[dict]) -> str:
    """Convert a sequence of {x,y,t} points into a u/d/l/r move string.

    Matches the classifier's training convention (u=-y, d=+y, l=-x, r=+x).
    Non-unit steps are decomposed (x first, then y) so the string is always
    valid u/d/l/r; in practice the game only emits unit orthogonal steps.
    """
    chars: list[str] = []
    for a, b in zip(pts, pts[1:]):
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        if dx > 0:
            chars.append("r" * dx)
        elif dx < 0:
            chars.append("l" * (-dx))
        if dy > 0:
            chars.append("d" * dy)
        elif dy < 0:
            chars.append("u" * (-dy))
    return "".join(chars)


def trace_metrics(pts: list[dict]) -> dict:
    """Per-trace metrics needed for the explorer's run cards (a subset of what
    analysis.R derives; both compute the same quantities the same way)."""
    points = len(pts)
    moves = points - 1
    cells = {(p["x"], p["y"]) for p in pts}
    unique = len(cells)
    revisits = points - unique
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    duration_ms = pts[-1]["t"] - pts[0]["t"] if points else 0
    # per-move timing: thinking (paused) vs execution (fluent)
    dts = [pts[i]["t"] - pts[i - 1]["t"] for i in range(1, points)]
    n_think = sum(1 for d in dts if d >= PAUSE_MS)
    return {
        "moves": moves,
        "unique_cells": unique,
        "revisits": revisits,
        "duration_s": round(duration_ms / 1000, 3),
        "thinking_moves": n_think,
        "thinking_frac": round(n_think / moves, 4) if moves > 0 else float("nan"),
        "longest_pause_ms": max(dts) if dts else 0,
        "bbox_w": max(xs) - min(xs) + 1 if xs else 0,
        "bbox_h": max(ys) - min(ys) + 1 if ys else 0,
    }


def main() -> None:
    print("Training path-shape classifier on", LABELLED_CSV.name, "...")
    model, evaluation = train_and_evaluate(LABELLED_CSV)
    print(f"  classifier holdout accuracy: {evaluation.accuracy:.3f}")

    print("Loading test rows ...")
    test = pd.read_csv(TEST_CSV)
    # drop the stray malformed 'level' row and anything without a parseable result
    test = test[test["level"].isin(LEVEL_ORDER)].copy()

    records = []
    move_strings = []
    move_index = []  # row position -> index into move_strings (or None)

    for _, row in test.reset_index(drop=True).iterrows():
        try:
            pts = json.loads(row["result"])
        except (TypeError, ValueError):
            continue
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        m = trace_metrics(pts)
        rec = {
            "trace_id": row["id"],
            "user_id": row["user_id"],
            "level": row["level"],
            "platform": row["platform"],
            "created_at": row["created_at"],
            **m,
            "pts": pts,  # kept for cohort export
        }
        ms = points_to_moves(pts)
        if ms:
            move_index.append(len(move_strings))
            move_strings.append(ms)
        else:
            move_index.append(None)
        records.append(rec)

    print(f"  parsed {len(records)} traces; classifying shapes ...")
    # batch-classify for speed
    labels = [None] * len(records)
    probs = [None] * len(records)
    valid = [(ri, mi) for ri, mi in enumerate(move_index) if mi is not None]
    if valid:
        texts = [move_strings[mi] for _, mi in valid]
        pred = model.predict(texts)
        proba = model.predict_proba(texts)
        classes = list(model.classes_)
        for (ri, _), lab, pr in zip(valid, pred, proba):
            labels[ri] = lab
            probs[ri] = {c: round(float(p), 4) for c, p in zip(classes, pr)}

    df = pd.DataFrame(records)
    df["pattern"] = labels
    df["pattern_conf"] = [max(p.values()) if p else float("nan") for p in probs]

    # join demographics
    users = pd.read_csv(USERS_CSV)[["user_id", *DEMO_COLS]]
    df = df.merge(users, on="user_id", how="left")

    # --- true optimality vs the exact Concorde optimal per level ---
    optima = level_optima()
    print("  optimal moves per level:", {k: optima[k] for k in LEVEL_ORDER if k in optima})
    df["optimal_moves"] = df["level"].map(optima).astype("Int64")
    moves_ok = df["moves"] > 0
    ratio = df["optimal_moves"].astype("float") / df["moves"]
    df["optimality"] = ratio.where(moves_ok).clip(upper=1.0).round(4)
    df["redundancy"] = (df["moves"] / df["optimal_moves"].astype("float")).where(moves_ok).round(4)

    # ordered categorical for nice plotting
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)

    # --- write cohort data for the interactive explorer ---
    # group runs under each user, attach demographics + computed pattern/metrics
    cohort_users = []
    for uid, g in df.groupby("user_id", sort=False):
        demo = {c: (None if pd.isna(g.iloc[0][c]) else g.iloc[0][c]) for c in DEMO_COLS}
        runs = []
        for _, r in g.iterrows():
            runs.append(
                {
                    "level": r["level"],
                    "trace_id": int(r["trace_id"]),
                    "platform": r["platform"],
                    "pattern": r["pattern"],
                    "pattern_conf": None if pd.isna(r["pattern_conf"]) else r["pattern_conf"],
                    "moves": int(r["moves"]),
                    "optimal_moves": None if pd.isna(r["optimal_moves"]) else int(r["optimal_moves"]),
                    "unique_cells": int(r["unique_cells"]),
                    "revisits": int(r["revisits"]),
                    "duration_s": r["duration_s"],
                    "thinking_moves": int(r["thinking_moves"]),
                    "thinking_frac": None if pd.isna(r["thinking_frac"]) else r["thinking_frac"],
                    "longest_pause_ms": int(r["longest_pause_ms"]),
                    "optimality": None if pd.isna(r["optimality"]) else r["optimality"],
                    "redundancy": None if pd.isna(r["redundancy"]) else r["redundancy"],
                    "W": int(r["bbox_w"]),
                    "H": int(r["bbox_h"]),
                    "pts": r["pts"],
                }
            )
        runs.sort(key=lambda x: LEVEL_ORDER.index(x["level"]) if x["level"] in LEVEL_ORDER else 99)
        cohort_users.append({"user_id": uid, **demo, "runs": runs})

    payload = {
        "levels": LEVEL_ORDER,
        "demo_cols": DEMO_COLS,
        "patterns": ["snake", "spiral", "random_walk"],
        "users": cohort_users,
    }
    OUT_COHORT.write_text(
        "window.COHORT=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    size_mb = OUT_COHORT.stat().st_size / 1e6
    print(f"Wrote {OUT_COHORT}  ({len(cohort_users)} users, {size_mb:.1f} MB)")

    # quick console summary
    print("\nPattern mix by level:")
    print(
        pd.crosstab(df["level"], df["pattern"], normalize="index")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
