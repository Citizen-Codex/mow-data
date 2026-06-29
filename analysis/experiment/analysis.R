# Mow-the-Lawn player-behaviour analysis - all static figures in one script.
#
# Inputs (from build_metrics.py / build_optimal.py):
#   trace_metrics.csv     one row per trace (metrics + demographics)
#   cell_aggregates.csv   per-cell visit-flow + pause stats, per round
#   modal_paths.csv       most common exact path per round (with geometry)
#   optimal_paths.csv     the exact Concorde optimal path per round (geometry)
# Output: figures/*.png  (+ summary stats printed to console)
#
# Optimality ("score") = optimal_moves / player_moves  (1.0 = optimal play),
# where optimal_moves is the exact Concorde minimum covering walk per level.
# A "thinking" move = wait >= 1s before the move (PAUSE_MS in build_metrics.py).
# Tutorial is guided, so it is dropped from optimality/scoring/pause figures.

library(tidyverse)
library(jsonlite)

out_dir <- "figures"
dir.create(out_dir, showWarnings = FALSE)
LEVELS  <- c("tutorial", "round1", "round2", "bonus1", "bonus2", "bonus3")
PALETTE <- c(snake = "#3ddc97", spiral = "#5b8cff", random_walk = "#ff5c8a")

save_fig <- function(p, name, w = 10, h = 6) {
  ggsave(file.path(out_dir, name), p, width = w, height = h, dpi = 130, bg = "white")
  message("wrote ", file.path(out_dir, name))
}
theme_mow <- theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(color = "grey35"),
        panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold"))
theme_grid <- theme(aspect.ratio = 1, panel.grid = element_blank(), axis.text = element_blank())

m <- read_csv("trace_metrics.csv", show_col_types = FALSE) %>%
  mutate(level = factor(level, levels = LEVELS, ordered = TRUE),
         pattern = factor(pattern, levels = c("snake", "spiral", "random_walk")))
# rounds used for optimality / scoring / pauses (exclude guided tutorial)
scored <- m %>% filter(level != "tutorial", is.finite(optimality))

# completed_all: did this user play *every* non-tutorial round? Per-user stats
# otherwise mix players who only did the easy early rounds with those who did
# all of them (only ~38% ever attempt a bonus round). Filtering to this flag
# gives a consistent pool for analyses where the set of rounds must be held
# fixed (e.g. Q10's per-user mean optimality, where the easy-round-only players
# pile up a false spike at optimality 1.0).
n_rounds_total <- n_distinct(scored$level)
scored <- scored %>%
  group_by(user_id) %>%
  mutate(completed_all = n_distinct(level) == n_rounds_total) %>%
  ungroup()
message(sprintf("completed_all: %d of %d users played all %d rounds",
                n_distinct(scored$user_id[scored$completed_all]),
                n_distinct(scored$user_id), n_rounds_total))

# ===========================================================================
# Q1. Path-shape mix by round
# ===========================================================================
mix <- m %>% filter(!is.na(pattern)) %>%
  count(level, pattern) %>% group_by(level) %>% mutate(prop = n / sum(n)) %>% ungroup()

save_fig(
  ggplot(mix, aes(level, prop, fill = pattern)) +
    geom_col(width = 0.8) +
    geom_text(aes(label = ifelse(prop > 0.06, scales::percent(prop, 1), "")),
              position = position_stack(vjust = 0.5), size = 3.2, color = "white") +
    scale_fill_manual(values = PALETTE) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "Path-shape mix shifts sharply across rounds",
         subtitle = "Spiral dominates early; snake takes over on the largest grids",
         x = NULL, y = "share of traces", fill = "pattern") +
    theme_mow,
  "q1_pattern_mix_by_round.png", 9, 5)

# ===========================================================================
# Q2. Time taken vs optimality, by round (does taking your time pay off?)
# ===========================================================================
# Optimality (not raw moves) is the skill measure: moves scale with grid size,
# so move counts only tell you the grid, not how well someone played. Cap time
# per round at this percentile to drop abandoned-device (huge idle time) traces.
# Adjust CAP_Q to tighten/loosen.
CAP_Q <- 0.99
q2d <- scored %>%
  filter(duration_s > 0) %>%
  group_by(level) %>%
  filter(duration_s <= quantile(duration_s, CAP_Q, na.rm = TRUE)) %>%
  ungroup()
message(sprintf("Q2: capped time at p%g per round, kept %d of %d traces (dropped %d outliers)",
                CAP_Q * 100, nrow(q2d), sum(scored$duration_s > 0),
                sum(scored$duration_s > 0) - nrow(q2d)))

q2_lab <- q2d %>% group_by(level) %>%
  summarise(rho = cor(duration_s, optimality, method = "spearman"),
            x = min(duration_s), y = max(optimality), .groups = "drop") %>%
  mutate(lab = sprintf("rho = %.2f", rho))

save_fig(
  ggplot(q2d, aes(duration_s, optimality)) +
    geom_point(aes(color = level), alpha = 0.25, size = 1, show.legend = FALSE) +
    geom_smooth(method = "loess", span = 0.9, se = TRUE,
                color = "grey20", fill = "grey60", alpha = 0.25, linewidth = 0.7) +
    geom_text(data = q2_lab, aes(x, y, label = lab),
              hjust = 0, vjust = 1, family = "mono", size = 3.4, color = "grey25") +
    facet_wrap(~level, scales = "free_x", ncol = 3) +
    scale_color_brewer(palette = "Set2") +
    scale_x_log10(labels = scales::comma) +
    coord_cartesian(ylim = c(0.6, 1)) +
    labs(title = "Time taken vs optimality, by round",
         subtitle = "Each point is a trace; loess fit with 95% CI. Within every round, players who spend longer score modestly higher (rho > 0).\nTime capped at the 99th percentile per round (drops abandoned-device traces). Log time axis.",
         x = "duration (s, log scale)", y = "optimality") +
    theme_mow,
  "q2_time_vs_optimality.png", 11, 7)

# ===========================================================================
# Q3. Top scorer vs everyone else (top optimality decile within each round)
# ===========================================================================
q3_tbl <- scored %>%
  group_by(level) %>%
  mutate(grp = if_else(optimality >= quantile(optimality, 0.9, na.rm = TRUE),
                       "Top 10%", "Everyone else")) %>%
  ungroup() %>%
  pivot_longer(c(optimality, revisits, ms_per_move), names_to = "metric", values_to = "value") %>%
  filter(is.finite(value)) %>%
  group_by(level, grp, metric) %>%
  summarise(med = median(value), .groups = "drop") %>%
  mutate(metric = recode(metric,
                         optimality = "optimality (higher = better)",
                         revisits = "revisited cells",
                         ms_per_move = "ms per move (pace)"))

save_fig(
  ggplot(q3_tbl, aes(level, med, fill = grp)) +
    geom_col(position = position_dodge(0.8), width = 0.75) +
    facet_wrap(~metric, scales = "free_y", ncol = 1) +
    scale_fill_manual(values = c("Top 10%" = "#f5b841", "Everyone else" = "#8497b6")) +
    labs(title = "Top scorers vs everyone else, by round",
         subtitle = "Medians. Top decile cover the lawn with far fewer revisits - and move no faster",
         x = NULL, y = NULL, fill = NULL) +
    theme_mow,
  "q3_top_vs_average.png", 9, 8)

# ===========================================================================
# Q4. Patterns + optimality by demographic group
# ===========================================================================
DEMOS <- c("age", "style", "gaming", "hand", "optimization")

eff_demo <- scored %>%
  pivot_longer(all_of(DEMOS), names_to = "demo", values_to = "group") %>% filter(!is.na(group))
save_fig(
  ggplot(eff_demo, aes(group, optimality)) +
    geom_boxplot(fill = "#5b8cff", alpha = 0.5, outlier.alpha = 0.08, linewidth = 0.4) +
    facet_wrap(~demo, scales = "free_x", ncol = 2) +
    coord_cartesian(ylim = c(0.6, 1)) +
    labs(title = "Optimality by demographic group",
         subtitle = "optimality = optimal moves / player moves;  1.0 = the Concorde optimum",
         x = NULL, y = "optimality") +
    theme_mow + theme(axis.text.x = element_text(angle = 25, hjust = 1, size = 8)),
  "q4a_optimality_by_demographic.png", 11, 8)

pat_demo <- m %>% filter(!is.na(pattern), level != "tutorial") %>%
  pivot_longer(all_of(DEMOS), names_to = "demo", values_to = "group") %>% filter(!is.na(group)) %>%
  count(demo, group, pattern) %>% group_by(demo, group) %>% mutate(prop = n / sum(n)) %>% ungroup()
save_fig(
  ggplot(pat_demo, aes(group, prop, fill = pattern)) +
    geom_col(width = 0.8) +
    facet_wrap(~demo, scales = "free_x", ncol = 2) +
    scale_fill_manual(values = PALETTE) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "Path-shape mix by demographic group", subtitle = "non-tutorial rounds",
         x = NULL, y = "share of traces", fill = "pattern") +
    theme_mow + theme(axis.text.x = element_text(angle = 25, hjust = 1, size = 8)),
  "q4b_patterns_by_demographic.png", 11, 8)

# ===========================================================================
# Q5. Did optimality improve between rounds?
# ===========================================================================
user_round <- scored %>% group_by(user_id, level) %>%
  summarise(optimality = mean(optimality), .groups = "drop")
trend <- user_round %>% group_by(level) %>%
  summarise(mean = mean(optimality), se = sd(optimality) / sqrt(n()), .groups = "drop")

save_fig(
  ggplot() +
    geom_line(data = user_round, aes(level, optimality, group = user_id), color = "grey70", alpha = 0.08) +
    geom_line(data = trend, aes(level, mean, group = 1), color = "#ff5c8a", linewidth = 1.2) +
    geom_pointrange(data = trend, aes(level, mean, ymin = mean - 1.96 * se, ymax = mean + 1.96 * se),
                    color = "#ff5c8a") +
    coord_cartesian(ylim = c(0.6, 1)) +
    labs(title = "Did optimality improve between rounds?",
         subtitle = "Faint lines = individual players; bold line = mean optimality with 95% CI",
         x = NULL, y = "optimality") +
    theme_mow,
  "q5_optimality_across_rounds.png", 9, 6)

paired <- user_round %>% filter(level %in% c("round1", "round2")) %>%
  pivot_wider(names_from = level, values_from = optimality) %>% drop_na()
if (nrow(paired) > 2) {
  tt <- t.test(paired$round2, paired$round1, paired = TRUE)
  message(sprintf("Paired round1->round2 optimality: dmean=%.4f, p=%.4g, n=%d",
                  mean(paired$round2 - paired$round1), tt$p.value, nrow(paired)))
}

# ===========================================================================
# Q6. The average path (visit-flow heatmap) + most common exact path
# ===========================================================================
cells <- read_csv("cell_aggregates.csv", show_col_types = FALSE) %>%
  mutate(level = factor(level, levels = LEVELS, ordered = TRUE))

save_fig(
  ggplot(filter(cells, trace_share >= 0.5), aes(x, y, fill = mean_step_frac)) +
    geom_tile() +
    facet_wrap(~level, scales = "free", ncol = 3) +
    scale_y_reverse() +
    scale_fill_viridis_c(option = "plasma", labels = scales::percent,
                         name = "avg point in\npath (0=start,\n1=end)") +
    labs(title = "The average path through each round",
         subtitle = "Colour = when, on average, players reach each cell. Following dark->bright traces the typical sweep. White gaps = obstacles.",
         x = NULL, y = NULL) +
    theme_mow + theme_grid,
  "q6_visit_heatmaps.png", 12, 8)

modal <- read_csv("modal_paths.csv", show_col_types = FALSE) %>%
  mutate(level = factor(level, levels = LEVELS, ordered = TRUE))
labs_df <- modal %>%
  mutate(lab = sprintf("%s\n%.0f%% of traces (%d of %d) · %d distinct routes",
                       level, 100 * modal_share, modal_count, n_traces, distinct_paths))
paths <- modal %>%
  mutate(path = map(path_json, ~ fromJSON(.x) %>% as_tibble())) %>%
  select(level, path) %>% unnest(path) %>%
  group_by(level) %>% mutate(step = row_number()) %>% ungroup() %>%
  left_join(select(labs_df, level, lab), by = "level") %>%
  mutate(lab = factor(lab, levels = labs_df$lab[order(match(labs_df$level, LEVELS))]))

save_fig(
  ggplot(paths, aes(x, y)) +
    geom_path(aes(color = step), linewidth = 1.1, lineend = "round") +
    geom_point(data = ~ filter(.x, step == 1), color = "#3ddc97", size = 2.6) +
    facet_wrap(~lab, scales = "free", ncol = 3) +
    scale_y_reverse() +
    scale_color_viridis_c(option = "viridis", name = "step") +
    labs(title = "The single most common exact path, by round",
         subtitle = "A dominant shared route exists only early (tutorial/round1); by the bonus rounds almost every route is unique.",
         x = NULL, y = NULL) +
    theme_mow + theme_grid + theme(strip.text = element_text(face = "bold", size = 9)),
  "q6_modal_paths.png", 12, 8)

# ---------------------------------------------------------------------------
# Q6c. The exact OPTIMAL path per round (Concorde) - the baseline that defines
#      optimality. Obstacles forcing revisits show up as the line crossing a
#      cell twice (e.g. bonus1's optimum needs 86 moves to cover 84 cells).
# ---------------------------------------------------------------------------
opt <- read_csv("optimal_paths.csv", show_col_types = FALSE) %>%
  mutate(level = factor(level, levels = LEVELS, ordered = TRUE))
opt_labs <- opt %>%
  mutate(lab = sprintf("%s\noptimal: %d moves to cover %d cells", level, optimal_moves, open_cells))
opt_paths <- opt %>%
  mutate(path = map(path_json, ~ fromJSON(.x) %>% as_tibble())) %>%
  select(level, path) %>% unnest(path) %>%
  group_by(level) %>% mutate(step = row_number()) %>% ungroup() %>%
  left_join(select(opt_labs, level, lab), by = "level") %>%
  mutate(lab = factor(lab, levels = opt_labs$lab[order(match(opt_labs$level, LEVELS))]))

save_fig(
  ggplot(opt_paths, aes(x, y)) +
    geom_path(aes(color = step), linewidth = 1.1, lineend = "round") +
    geom_point(data = ~ filter(.x, step == 1), color = "#3ddc97", size = 2.6) +
    facet_wrap(~lab, scales = "free", ncol = 3) +
    scale_y_reverse() +
    scale_color_viridis_c(option = "magma", name = "step") +
    labs(title = "The exact optimal path, by round (Concorde solver)",
         subtitle = "Minimum-length covering walk from the top-left start. This is the baseline optimality is scored against (optimality = these moves / player moves).",
         x = NULL, y = NULL) +
    theme_mow + theme_grid + theme(strip.text = element_text(face = "bold", size = 9)),
  "q6c_optimal_paths.png", 12, 8)

# ===========================================================================
# Q7. Where players stop to think (pause-rate heatmap)
# ===========================================================================
save_fig(
  ggplot(filter(cells, is.finite(pause_rate)), aes(x, y, fill = pause_rate)) +
    geom_tile() +
    facet_wrap(~level, scales = "free", ncol = 3) +
    scale_y_reverse() +
    # 99% of cells are <=30%; only the start corner spikes to ~90%, so cap the
    # scale at 30% (outliers squished to the top) to reveal the bulk variation.
    scale_fill_viridis_c(option = "mako", direction = -1, labels = scales::percent,
                         limits = c(0, 0.3), oob = scales::squish,
                         name = "chance a visit\ntriggers a pause\n(capped at 30%)") +
    labs(title = "Where players stop to think",
         subtitle = "Per cell: share of visits followed by a >=1s pause. The start corner (>=90%, off-scale) is the big planning moment; dead-ends and turns light up next.",
         x = NULL, y = NULL) +
    theme_mow + theme_grid,
  "q7_pause_heatmaps.png", 12, 8)

# ===========================================================================
# Q8. Thinking vs execution: top scorers (top 5% optimality, per round) vs rest.
#     Rows = share of MOVES / share of TIME; columns = the two groups.
# ===========================================================================
split <- scored %>%
  group_by(level) %>%
  mutate(grp = if_else(optimality >= quantile(optimality, 0.95, na.rm = TRUE),
                       "Top 5%", "Everyone else")) %>%
  group_by(level, grp) %>%
  summarise(`moves: thinking`  = sum(thinking_moves) / sum(moves),
            `moves: execution` = 1 - sum(thinking_moves) / sum(moves),
            `time: thinking`   = sum(thinking_ms) / sum(thinking_ms + execution_ms),
            `time: execution`  = sum(execution_ms) / sum(thinking_ms + execution_ms),
            .groups = "drop") %>%
  pivot_longer(c(-level, -grp), names_to = "kv", values_to = "share") %>%
  separate(kv, c("facet", "kind"), sep = ": ") %>%
  mutate(kind = factor(kind, levels = c("execution", "thinking")),
         facet = recode(facet, moves = "share of MOVES", time = "share of TIME"),
         grp = factor(grp, levels = c("Top 5%", "Everyone else")))

save_fig(
  ggplot(split, aes(level, share, fill = kind)) +
    geom_col(width = 0.8) +
    geom_text(aes(label = ifelse(share > 0.06, scales::percent(share, 1), "")),
              position = position_stack(vjust = 0.5), size = 2.9, color = "white") +
    facet_grid(facet ~ grp) +
    scale_fill_manual(values = c(execution = "#5b8cff", thinking = "#f5b841")) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "Thinking vs execution: top scorers vs everyone else",
         subtitle = "Top 5% = highest optimality within each round. A \"thinking\" move is one preceded by a wait of >=1 second.\nShare of MOVES = fraction of all moves that were thinking vs fluent. Share of TIME = fraction of total time spent in each.",
         x = NULL, y = NULL, fill = NULL) +
    theme_mow + theme(axis.text.x = element_text(angle = 20, hjust = 1, size = 8),
                      strip.text.y = element_text(angle = -90)),
  "q8_think_vs_exec.png", 10, 7)

# ===========================================================================
# Q9. Does thinking pay off?
#   No binning - a loess fit + 95% CI over the raw traces uses all the data and
#   the band widens where points are sparse, so noisy regions show as wide CI
#   instead of misleading bin means.
# ===========================================================================
save_fig(
  ggplot(scored, aes(thinking_frac, optimality)) +
    geom_point(alpha = 0.05, size = 0.6, color = "#5b8cff") +
    geom_smooth(method = "loess", span = 0.9, se = TRUE,
                color = "#ff5c8a", fill = "#ff5c8a", alpha = 0.2, linewidth = 0.9) +
    facet_wrap(~level, ncol = 3) +
    coord_cartesian(xlim = c(0, 0.6), ylim = c(0.75, 1)) +   # display window; fit uses all data
    labs(title = "Does thinking pay off?",
         subtitle = "Loess fit with 95% CI over raw traces (no binning). Points = traces; the band widens where data is sparse (high pause shares), so the noisy tail reads as uncertain rather than as a real dip.",
         x = "share of moves that are thinking pauses", y = "optimality") +
    theme_mow,
  "q9_thinking_vs_optimality.png", 11, 7)

# ===========================================================================
# Q10. Score distribution ACROSS USERS (each user's mean optimality over their
#      rounds). Restricted to players who completed EVERY round (completed_all):
#      per-user averaging only cancels the round-difficulty confound when the
#      set of rounds is held fixed. Without this filter the ~62% of players who
#      stop after the easy early rounds average to ~1.0 and pile up a false
#      "perfect player" spike at optimality 1.0 (round1's optimum is trivial).
# ===========================================================================
user_eff <- scored %>%
  filter(completed_all) %>%
  group_by(user_id) %>%
  summarise(optimality = mean(optimality), n_rounds = n(), .groups = "drop")
md <- median(user_eff$optimality)

save_fig(
  ggplot(user_eff, aes(optimality)) +
    geom_histogram(binwidth = 0.005, fill = "#5b8cff", color = "white", linewidth = 0.2) +
    geom_vline(xintercept = md, linetype = "dashed", color = "#ff5c8a", linewidth = 0.8) +
    annotate("text", x = md, y = Inf, label = sprintf("  median %.3f", md),
             hjust = 0, vjust = 1.5, color = "#ff5c8a", family = "mono", size = 3.6) +
    scale_x_continuous(limits = c(0.5, 1.02)) +
    labs(title = "Player skill: distribution of average optimality per user",
         subtitle = sprintf("One value per player = mean optimality across all %d rounds, restricted to players who completed every round (n = %s players).\nScore = optimal moves / player moves; 1.0 = the Concorde optimum.", n_rounds_total, scales::comma(nrow(user_eff))),
         x = "average optimality per user", y = "players") +
    theme_mow,
  "q10_score_hist_overall.png", 10, 5.5)

meds <- scored %>% group_by(level) %>% summarise(md = median(optimality), .groups = "drop")
save_fig(
  ggplot(scored, aes(optimality)) +
    geom_histogram(aes(fill = level), binwidth = 0.02, color = "white", linewidth = 0.15, show.legend = FALSE) +
    geom_vline(data = meds, aes(xintercept = md), linetype = "dashed", color = "grey20") +
    facet_wrap(~level, scales = "free_y", ncol = 2) +
    scale_fill_brewer(palette = "Set2") +
    scale_x_continuous(limits = c(0.5, 1.02)) +
    labs(title = "Score distribution by round",
         subtitle = "Dashed line = round median. Spread widens on the larger bonus grids.",
         x = "optimality", y = "traces") +
    theme_mow,
  "q11_score_hist_by_round.png", 10, 7)

# ===========================================================================
# Q12. Core metric distributions by round
# ===========================================================================
long <- scored %>% filter(duration_s > 0) %>%
  transmute(level, `total moves` = moves, `revisited cells` = revisits,
            `duration (log10 s)` = log10(duration_s), `thinking-move share` = thinking_frac) %>%
  pivot_longer(-level, names_to = "metric", values_to = "value")
save_fig(
  ggplot(long, aes(level, value, fill = level)) +
    geom_violin(scale = "width", color = NA, alpha = 0.85, show.legend = FALSE) +
    geom_boxplot(width = 0.14, outlier.shape = NA, fill = "white", alpha = 0.7) +
    facet_wrap(~metric, scales = "free_y") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Core metric distributions by round",
         subtitle = "Violin = full distribution; box = IQR + median. Duration is log10 seconds (idle-pause outliers).",
         x = NULL, y = NULL) +
    theme_mow + theme(axis.text.x = element_text(angle = 20, hjust = 1, size = 9)),
  "q12_metric_distributions.png", 11, 7)

# ===========================================================================
# Summary stats -> console
# ===========================================================================
by_round <- m %>% group_by(level) %>%
  summarise(n = n(), players = n_distinct(user_id),
            eff_mean = mean(optimality, na.rm = TRUE), eff_median = median(optimality, na.rm = TRUE),
            eff_sd = sd(optimality, na.rm = TRUE), moves_median = median(moves),
            revisits_median = median(revisits), duration_s_median = median(duration_s),
            thinking_frac_median = median(thinking_frac, na.rm = TRUE), .groups = "drop") %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))
message("\nSummary by round:")
print(by_round)
message("\nDone. Figures in ./", out_dir, "/")
