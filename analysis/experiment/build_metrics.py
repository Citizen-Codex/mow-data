"""Build a tidy per-trace metrics table from the Mow-the-Lawn experiment data.

Reads the raw experiment CSVs (test rows = path traces, user rows = demographics),
derives per-trace metrics, labels each path's shape with the existing path
classifier, joins demographics, and writes:

  - trace_metrics.csv   one row per trace, ready for static analysis / ggplot
  - cohort_data.js      window.COHORT = {...} for the interactive cohort explorer

Optimality here is the self-contained proxy chosen for v1:
    redundancy = moves / (unique_cells - 1)      (1.0 = no backtracking, >1 worse)
    optimality = 1 / redundancy                  (1.0 best, ->0 worse)
No grid/solver baseline is needed; it measures how much a player retraced
their own covered cells. Swap in a true-optimal baseline later if desired.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # analysis/experiment -> repo root
sys.path.insert(0, str(REPO))

from classifier import train_and_evaluate  # noqa: E402

TEST_CSV = HERE / "mow_test_rows.csv"
USERS_CSV = HERE / "mow_users_rows.csv"
LABELLED_CSV = REPO / "data" / "labelled_paths.csv"

OUT_METRICS = HERE / "trace_metrics.csv"
OUT_COHORT = HERE / "cohort_data.js"
OUT_CELLS = HERE / "cell_aggregates.csv"   # per-cell visit-flow + pause stats
OUT_MODAL = HERE / "modal_paths.csv"

# Logical round ordering (difficulty / grid size grows along this axis).
LEVEL_ORDER = ["tutorial", "round1", "round2", "bonus1", "bonus2", "bonus3"]

DEMO_COLS = ["age", "style", "gaming", "hand", "optimization"]

# A move that took longer than this (ms) to make is a deliberate "thinking"
# move; faster ones are fluent "execution". Matches the 1000ms reference line
# already used in eda.R. Adjust here to re-cut the thinking/execution split.
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
    points = len(pts)
    moves = points - 1
    cells = {(p["x"], p["y"]) for p in pts}
    unique = len(cells)
    revisits = points - unique
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    duration_ms = pts[-1]["t"] - pts[0]["t"] if points else 0
    # optimality proxy
    denom = unique - 1
    redundancy = moves / denom if denom > 0 else float("nan")
    optimality = denom / moves if moves > 0 else float("nan")
    # per-move timing: thinking (paused) vs execution (fluent)
    dts = [pts[i]["t"] - pts[i - 1]["t"] for i in range(1, points)]
    thinking = [d for d in dts if d >= PAUSE_MS]
    n_think = len(thinking)
    think_ms = sum(thinking)
    exec_ms = sum(d for d in dts if d < PAUSE_MS)
    med_dt = sorted(dts)[len(dts) // 2] if dts else float("nan")
    return {
        "points": points,
        "moves": moves,
        "unique_cells": unique,
        "revisits": revisits,
        "duration_ms": duration_ms,
        "duration_s": round(duration_ms / 1000, 3),
        "ms_per_move": round(duration_ms / moves, 1) if moves > 0 else float("nan"),
        "median_dt_ms": med_dt,
        "thinking_moves": n_think,
        "thinking_frac": round(n_think / moves, 4) if moves > 0 else float("nan"),
        "thinking_ms": think_ms,
        "execution_ms": exec_ms,
        "thinking_time_frac": round(think_ms / duration_ms, 4) if duration_ms > 0 else float("nan"),
        "longest_pause_ms": max(dts) if dts else 0,
        "bbox_w": max(xs) - min(xs) + 1 if xs else 0,
        "bbox_h": max(ys) - min(ys) + 1 if ys else 0,
        "redundancy": round(redundancy, 4) if redundancy == redundancy else redundancy,
        "optimality": round(optimality, 4) if optimality == optimality else optimality,
    }


def write_aggregates(df: pd.DataFrame) -> None:
    """Per-level cell-visit heatmap + the single most common exact path."""
    from collections import Counter, defaultdict

    cell_rows = []
    modal_rows = []
    for level in LEVEL_ORDER:
        g = df[df["level"] == level]
        if g.empty:
            continue
        n_traces = len(g)
        # grid extent for this level (max over traces)
        gw = int(g["bbox_w"].max())
        gh = int(g["bbox_h"].max())

        visits = defaultdict(int)      # (x,y) -> total times stepped on
        touched = defaultdict(int)     # (x,y) -> number of traces that touched it
        step_sum = defaultdict(float)  # (x,y) -> sum of normalised step position (0..1)
        pause_at = defaultdict(int)    # (x,y) -> # thinking-pauses that happened here
        pause_ms_at = defaultdict(float)  # (x,y) -> total ms paused here
        sig_counter: Counter = Counter()   # exact move-signature -> count
        sig_example: dict[str, list] = {}  # signature -> representative pts

        for pts in g["pts"]:
            cells_here = {}
            last = len(pts) - 1 or 1
            for i, p in enumerate(pts):
                c = (p["x"], p["y"])
                visits[c] += 1
                # first time this trace reaches the cell defines its arrival order
                if c not in cells_here:
                    cells_here[c] = i / last
                # a pause is the wait *before* the next move, i.e. while sitting at c
                if i + 1 < len(pts):
                    dt = pts[i + 1]["t"] - p["t"]
                    if dt >= PAUSE_MS:
                        pause_at[c] += 1
                        pause_ms_at[c] += dt
            for c, frac in cells_here.items():
                touched[c] += 1
                step_sum[c] += frac
            sig = points_to_moves(pts)
            sig_counter[sig] += 1
            if sig not in sig_example:
                sig_example[sig] = pts

        for (x, y) in sorted(set(visits) | set(touched)):
            t = touched[(x, y)]
            v = visits[(x, y)]
            pa = pause_at[(x, y)]
            cell_rows.append(
                {
                    "level": level,
                    "x": x,
                    "y": y,
                    "grid_w": gw,
                    "grid_h": gh,
                    "visits": v,
                    "traces_touching": t,
                    "trace_share": round(t / n_traces, 4),
                    # avg point in the path (0=start,1=end) when the cell is first reached
                    "mean_step_frac": round(step_sum[(x, y)] / t, 4) if t else float("nan"),
                    "pauses": pa,
                    "pauses_per_trace": round(pa / n_traces, 4),
                    # probability that landing on this cell triggers a think
                    "pause_rate": round(pa / v, 4) if v else float("nan"),
                    "mean_pause_ms": round(pause_ms_at[(x, y)] / pa, 1) if pa else 0.0,
                }
            )

        sig, count = sig_counter.most_common(1)[0]
        modal_rows.append(
            {
                "level": level,
                "n_traces": n_traces,
                "modal_count": count,
                "modal_share": round(count / n_traces, 4),
                "distinct_paths": len(sig_counter),
                "grid_w": gw,
                "grid_h": gh,
                "path_json": json.dumps(
                    [{"x": p["x"], "y": p["y"]} for p in sig_example[sig]],
                    separators=(",", ":"),
                ),
            }
        )

    pd.DataFrame(cell_rows).to_csv(OUT_CELLS, index=False)
    print(f"Wrote {OUT_CELLS}  ({len(cell_rows)} cells)")
    md = pd.DataFrame(modal_rows)
    md.to_csv(OUT_MODAL, index=False)
    print(f"Wrote {OUT_MODAL}  ({len(modal_rows)} levels)")
    print("\nMost common exact path per level:")
    print(
        md[["level", "n_traces", "distinct_paths", "modal_count", "modal_share"]]
        .to_string(index=False)
    )


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
            "pts": pts,  # kept for cohort export; dropped before CSV
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

    # ordered categorical for nice plotting
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)

    # --- write tidy CSV (no pts / probs columns) ---
    csv_df = df.drop(columns=["pts"])
    csv_df.to_csv(OUT_METRICS, index=False)
    print(f"Wrote {OUT_METRICS}  ({len(csv_df)} rows, {len(csv_df.columns)} cols)")

    # --- per-level heatmap + most common path aggregates ---
    write_aggregates(df)

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
