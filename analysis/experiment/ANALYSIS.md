# Mow-the-Lawn player-behaviour analysis

Analysis of human play traces (`mow_test_rows.csv`) joined to demographics
(`mow_users_rows.csv`). Each trace is a timestamped `{x,y,t}` path covering a
grid; rounds grow from 6×6 (tutorial/round1) to 14×14 (bonus3).

## Approach: static figures **and** an interactive explorer

- **Static ggplot figures** answer the fixed statistical questions (correlations,
  group comparisons, trends) — reproducible, report-ready.
- **An interactive cohort explorer** handles the *spatial* questions ("do these
  shapes really look like snakes/spirals?") that a bar chart can't.

## Pipeline

```
build_metrics.py   ->  trace_metrics.csv      one row per trace (metrics + demographics)
                       cell_aggregates.csv    per-cell visit-flow + pause stats, per round
                       modal_paths.csv        most common exact path per round (with geometry)
                       cohort_data.js         window.COHORT for the explorer
analysis.R         ->  figures/q1..q12*.png   all static figures; summary stats -> console
cohort_explorer.html   open in a browser; reads cohort_data.js
```

Three derived CSVs feed everything: `trace_metrics.csv` (tidy per-trace table),
`cell_aggregates.csv` (per-cell heatmap inputs), `modal_paths.csv` (modal-path
geometry). Summary stats print to the console rather than to disk.

### Running

A scientific Python env (pandas, scikit-learn) and R (tidyverse, jsonlite) are
required. The project normally uses `uv`; any env with those libs works.

```bash
python build_metrics.py     # regenerate all derived data
Rscript analysis.R          # all figures q1-q12 + summary stats to console
open cohort_explorer.html   # interactive
```

## Metric definitions

- **pattern** — snake / spiral / random_walk, from the project's `classifier.py`
  (char-ngram logistic regression; 0.97 holdout accuracy) applied to each path's
  u/d/l/r move string.
- **optimality** = `(unique_cells - 1) / moves`. 1.0 = covered the lawn with no
  backtracking; lower = more retracing. This is the self-contained optimality
  proxy (no grid/solver baseline needed). `redundancy = 1/optimality`.
- **thinking move** — a move preceded by a wait ≥ `PAUSE_MS` (1000 ms, set in
  `build_metrics.py`); the rest are fluent **execution** moves. `thinking_frac` is
  the share of such moves; `thinking_time_frac` is their share of total time.
- Tutorial is a *guided* level, so it is excluded from optimality/scoring figures
  (kept only for the pattern-mix and average-path views).

## Key findings

1. **Path shape shifts sharply by round.** round1 is 82% spiral; by bonus3 it's
   87% snake. As grids grow, players abandon spirals for boustrophedon snaking.
   round2 is the noisy transition point (41% random_walk).
2. **Time vs moves is a Simpson's paradox.** Pooled, it looks positive
   (Spearman ρ ≈ 0.48) — but that's purely grid size (bigger round → more moves
   AND more time). Faceted *within* each round the correlation flips **negative**
   (ρ ≈ −0.14 to −0.31): players who make more moves are clicking faster, while
   careful players make fewer moves but pause to deliberate. Always read this one
   per-round. (Pearson is also wrecked by a few multi-hour idle pauses.)
3. **Top scorers backtrack, they don't rush.** The top optimality decile covers
   the lawn with far fewer revisits but moves at the *same or slower* pace —
   deliberation beats speed.
4. **Demographics:** clear age gradient (younger more efficient; 60+ lowest);
   self-described "I follow a set pattern" players are slightly more efficient;
   regular gamers marginally better; handedness shows no effect.
5. **Optimality does not improve across rounds** — it's roughly flat and dips
   slightly from round1→round2 (paired Δ ≈ -0.006, p≈1e-5) as grids get harder.
   No within-session learning curve.
6. **The "average path":** everyone starts top-left and sweeps the perimeter
   first, reaching centres last (spiral-inward tendency, clearest on bonus3). A
   single *dominant* route only exists early — tutorial 35%, round1 24%,
   round2 13% — by the bonus rounds essentially every route is unique.
7. **Pauses cluster at the start.** A "thinking" move = >1s wait before moving
   (`PAUSE_MS`). The start corner overwhelmingly lights up (~90% of starts get a
   pause, far off-scale vs the ~11% median cell): players plan the whole route up
   front. Secondary hotspots sit near obstacles, turns, and dead-ends.
   Bulk-interior sweep cells rarely trigger a pause.
8. **A few moves eat all the time.** Thinking moves are a small minority of moves
   (round1 23% → bonus3 11%) but consume the *majority of the clock*
   (57–93% of each round's time). Players get more fluent (lower thinking share)
   on bigger grids, but the time is still front-loaded into a handful of pauses.
9. **Thinking has diminishing returns** (`q9`): a little deliberation tracks with
   higher optimality — a robust, steep rise from 0 to ~15–20% pause share (tight
   95% CI), then a plateau. Plotted as a loess fit + CI over raw traces (not
   bins); the band fans out at high pause shares where data is sparse, so the flat
   tail is genuinely uncertain rather than a real decline. Correlational, not causal.
10. **Player skill is left-skewed** (`q10`, per-user mean optimality, median ≈ 0.92):
    most players are consistently near-optimal, with a long tail of heavy
    backtrackers. Averaging per user (not per trace) avoids reading round
    difficulty as skill — some rounds force overlaps. Per round (`q11`) the spread
    widens on the larger bonus grids; bonus1 is the hardest (median 0.87). The
    full by-round table prints to the console when you run `analysis.R`.

## Interactive explorer

`cohort_explorer.html` — filter the full set of 15k traces by round, shape, and
any demographic; the gallery draws each path (revisited cells flagged), and
clicking a tile opens an animated replay with per-trace stats. Use it to
sanity-check the classifier's labels and to eyeball how a filter's paths differ.

Three tabs:
- **traces** — the cohort gallery. Two views (top-left **view** selector):
  - *every trace* — one tile per trace.
  - *distinct solutions* — collapses identical exact paths into one tile showing
    `×count` and the % of (filtered) traces that took it; sort by optimality or by
    **popularity**. Interactive companion to the modal-path figure: e.g. round1
    has 616 distinct routes, top one done by 924 players (24%); bonus3 is all
    unique. Combine with a demographic filter for the popular route *within* a group.
- **round maps** — live-rendered, recomputed from every trace's
  timing in the browser. Pick a round; the **pause-threshold slider** sets how many
  seconds counts as a "thinking" pause and redraws the pause-rate heatmap (the
  interactive version of `q7`) plus the average-path flow heatmap (`q6`). At 1s the
  pause map matches the static figure exactly; the live "% of moves that are pauses"
  readout updates with the slider. Choose **all rounds** in the round selector for a
  6-up small-multiples grid, with a metric toggle (pause-rate or average path)
  so you can compare every round side by side.
- **report figures** — the static statistical figures (q1–q5, q8–q12) embedded as
  images for one-stop viewing.
