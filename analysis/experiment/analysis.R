# Mow-the-Lawn player-behaviour analysis - all static figures in one script.
#
# Self-contained: every metric is derived in R straight from the raw experiment
# data, so there are no Python-built intermediate CSVs to keep in sync. Inputs:
#   mow_test_rows.csv    raw traces (result = JSON [{x,y,t}, ...] per play)
#   mow_users_rows.csv   demographics, one row per user
#   optimal_paths.csv    exact Concorde optimal path + move count per round
#                        (the one Python-produced input - Concorde can't run in
#                         R; regenerate with build_optimal.py when levels change)
# Output: figures/*.png  (+ summary stats printed to console)
#
# Optimality ("score") = optimal_moves / player_moves  (1.0 = optimal play),
# where optimal_moves is the exact Concorde minimum covering walk per level.
# A "thinking" move = wait >= 1s before the move (PAUSE_MS below).
# Tutorial is guided, so it is dropped from optimality/scoring/pause figures.
#
# NOTE: the path-shape classification figures (q1 pattern mix, q4b patterns by
# demographic) are no longer produced here - the classifier is a Python-only
# concern and those figures are frozen. Their existing PNGs are left untouched.

library(tidyverse)
library(jsonlite)

out_dir <- "figures"
dir.create(out_dir, showWarnings = FALSE)
LEVELS  <- c("tutorial", "round1", "round2", "bonus1", "bonus2", "bonus3")
DEMOS   <- c("age", "style", "gaming", "hand", "optimization")
PAUSE_MS <- 1000   # wait >= this (ms) before a move = a deliberate "thinking" move

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

# ===========================================================================
# Load raw data and derive every metric in R (replaces the Python CSVs).
# ===========================================================================
optimal_raw <- read_csv("optimal_paths.csv", show_col_types = FALSE)
users <- read_csv("mow_users_rows.csv", show_col_types = FALSE) %>%
  select(user_id, all_of(DEMOS))

# --- parse each trace's {x,y,t} JSON into a long point table -----------------
# u=-y, d=+y, l=-x, r=+x (game convention); t is elapsed ms since trace start.
safe_parse <- function(s) {
  out <- tryCatch(fromJSON(s), error = function(e) NULL)
  if (is.data.frame(out) && nrow(out) >= 2 && all(c("x", "y", "t") %in% names(out)))
    tibble(x = as.integer(out$x), y = as.integer(out$y), t = as.numeric(out$t))
  else NULL
}

message("parsing raw traces ...")
raw <- read_csv("mow_test_rows.csv", show_col_types = FALSE) %>%
  filter(level %in% LEVELS)
parsed <- raw %>%
  transmute(trace_id = id, user_id, level, platform, created_at,
            P = map(result, safe_parse)) %>%
  filter(!map_lgl(P, is.null))
trace_info <- parsed %>% select(trace_id, user_id, level, platform, created_at)

# one row per point; dt = wait *before* this point's move (t - prev t),
# dt_after = wait *after* sitting here (next t - t, used for per-cell pauses).
pts <- parsed %>%
  select(trace_id, level, P) %>%
  unnest(P) %>%
  group_by(trace_id) %>%
  mutate(i = row_number(), npts = n(),
         dt = t - lag(t), dt_after = lead(t) - t) %>%
  ungroup()
message(sprintf("  %s traces, %s points",
                scales::comma(n_distinct(pts$trace_id)), scales::comma(nrow(pts))))

# --- per-trace metrics -------------------------------------------------------
m <- pts %>%
  group_by(trace_id) %>%
  summarise(
    level        = first(level),
    points       = n(),
    moves        = points - 1,
    unique_cells = n_distinct(paste(x, y)),
    revisits     = points - unique_cells,
    duration_ms  = last(t) - first(t),
    duration_s   = round(duration_ms / 1000, 3),
    ms_per_move  = if_else(moves > 0, round(duration_ms / moves, 1), NA_real_),
    thinking_moves     = sum(dt >= PAUSE_MS, na.rm = TRUE),
    thinking_frac      = if_else(moves > 0, round(thinking_moves / moves, 4), NA_real_),
    thinking_ms        = sum(dt[dt >= PAUSE_MS], na.rm = TRUE),
    execution_ms       = sum(dt[dt <  PAUSE_MS], na.rm = TRUE),
    thinking_time_frac = if_else(duration_ms > 0, round(thinking_ms / duration_ms, 4), NA_real_),
    longest_pause_ms   = if_else(any(!is.na(dt)), max(dt, na.rm = TRUE), 0),
    bbox_w = max(x) - min(x) + 1,
    bbox_h = max(y) - min(y) + 1,
    .groups = "drop") %>%
  left_join(trace_info %>% select(trace_id, user_id, platform, created_at), by = "trace_id") %>%
  left_join(select(optimal_raw, level, optimal_moves), by = "level") %>%
  mutate(optimality = round(pmin(optimal_moves / moves, 1.0), 4),
         redundancy = round(moves / optimal_moves, 4)) %>%
  left_join(users, by = "user_id") %>%
  mutate(level = factor(level, levels = LEVELS, ordered = TRUE))

# rounds used for optimality / scoring / pauses (exclude guided tutorial)
scored <- m %>% filter(level != "tutorial", is.finite(optimality))

# ---------------------------------------------------------------------------
# Behavioural cohorts: segment players by *how they played*, an alternative to
# the self-reported demographics. Each is a user-level flag joined onto every
# trace, so any figure can facet/filter by cohort (see COHORTS).
#   completed_all - played every non-tutorial round: deeply engaged, deliberate.
#                   Also the consistent pool (fixed round set) used wherever a
#                   per-user mean would otherwise be confounded by *which* rounds
#                   a player attempted - e.g. Q10's false optimality-1.0 spike.
#   top/worst_solver  - top/bottom 10% by mean optimality,  *within completed_all*
#   quick/slow_solver - fastest/slowest 10% by mean solve time, *within completed_all*
# All tails are taken inside completed_all so the round set is held fixed -
# otherwise easy-round-only players dominate (mean optimality piles up at 1.0
# and their short easy rounds look "quick"). opt_pct/dur_pct = each user's
# percentile rank within the pool (fraction at or below).
COHORTS  <- c("completed_all", "top_solver", "worst_solver", "quick_solver", "slow_solver")
SOLVER_COHORTS <- c("top_solver", "worst_solver", "quick_solver", "slow_solver")  # the 4 tails
SOLVER_LABELS  <- c("top", "worst", "quick", "slow")  # short facet labels, same order
COHORT_Q <- 0.10
n_rounds_total <- n_distinct(scored$level)

user_agg <- scored %>%
  group_by(user_id) %>%
  summarise(completed_all = n_distinct(level) == n_rounds_total,
            mean_opt = mean(optimality),
            mean_dur = mean(duration_s[duration_s > 0]),  # ignore 0s; avg solve time
            .groups = "drop")

# all cohort thresholds taken only over the completed-all pool (fixed round set)
ca     <- user_agg %>% filter(completed_all)
opt_hi <- quantile(ca$mean_opt, 1 - COHORT_Q, na.rm = TRUE)
opt_lo <- quantile(ca$mean_opt, COHORT_Q, na.rm = TRUE)
ca_dur <- ca$mean_dur[is.finite(ca$mean_dur)]
dur_lo <- quantile(ca_dur, COHORT_Q, na.rm = TRUE)
dur_hi <- quantile(ca_dur, 1 - COHORT_Q, na.rm = TRUE)
opt_ecdf <- ecdf(ca$mean_opt)   # fraction of pool at or below -> percentile rank
dur_ecdf <- ecdf(ca_dur)

user_stats <- user_agg %>%
  mutate(opt_pct = if_else(completed_all, round(opt_ecdf(mean_opt), 4), NA_real_),
         dur_pct = if_else(completed_all & is.finite(mean_dur),
                           round(dur_ecdf(mean_dur), 4), NA_real_),
         top_solver   = completed_all & mean_opt >= opt_hi,
         worst_solver = completed_all & mean_opt <= opt_lo,
         quick_solver = completed_all & is.finite(mean_dur) & mean_dur <= dur_lo,
         slow_solver  = completed_all & is.finite(mean_dur) & mean_dur >= dur_hi)

scored <- scored %>%
  left_join(select(user_stats, user_id, all_of(COHORTS), opt_pct, dur_pct),
            by = "user_id")

message("cohort sizes:")
for (cset in COHORTS)
  message(sprintf("  %-13s %5d of %d users", cset,
                  sum(user_stats[[cset]], na.rm = TRUE), nrow(user_stats)))

# --- per-cell visit-flow + pause stats (reused for overall + per-cohort) -----
# Takes a long point table (optionally a cohort subset) -> one row per cell:
#   visits          total times any trace stepped on the cell
#   traces_touching distinct traces that reached it; trace_share = / n_traces
#   mean_step_frac  avg point-in-path (0=start,1=end) when first reached
#   pauses          # of >=1s waits that happened while sitting on the cell
#   pause_rate      share of visits followed by such a pause; mean_pause_ms its size
cell_aggregate <- function(p) {
  n_tr <- p %>% distinct(level, trace_id) %>% count(level, name = "n_traces")
  vis <- p %>%
    group_by(level, x, y) %>%
    summarise(visits   = n(),
              pauses   = sum(dt_after >= PAUSE_MS, na.rm = TRUE),
              pause_ms = sum(dt_after[dt_after >= PAUSE_MS], na.rm = TRUE),
              .groups = "drop")
  arr <- p %>%
    group_by(level, trace_id, x, y) %>%
    summarise(frac = (min(i) - 1) / pmax(first(npts) - 1, 1), .groups = "drop") %>%
    group_by(level, x, y) %>%
    summarise(traces_touching = n(), step_sum = sum(frac), .groups = "drop")
  vis %>%
    left_join(arr, by = c("level", "x", "y")) %>%
    left_join(n_tr, by = "level") %>%
    mutate(trace_share      = round(traces_touching / n_traces, 4),
           mean_step_frac   = round(step_sum / traces_touching, 4),
           pauses_per_trace = round(pauses / n_traces, 4),
           pause_rate       = if_else(visits > 0, round(pauses / visits, 4), NA_real_),
           mean_pause_ms    = if_else(pauses > 0, round(pause_ms / pauses, 1), 0),
           level = factor(level, levels = LEVELS, ordered = TRUE))
}

cells <- cell_aggregate(pts)

# the same aggregates restricted to each solver cohort (granular Q6/Q7)
cells_co <- map_dfr(SOLVER_COHORTS, function(co) {
  uids <- user_stats$user_id[user_stats[[co]] %in% TRUE]
  tids <- trace_info$trace_id[trace_info$user_id %in% uids]
  sub  <- filter(pts, trace_id %in% tids)
  if (nrow(sub) == 0) return(NULL)
  cell_aggregate(sub) %>% mutate(cohort = co)
}) %>%
  mutate(cohort = factor(cohort, levels = SOLVER_COHORTS, labels = SOLVER_LABELS))

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
# Q4. Optimality by demographic group (+ significance test)
# ===========================================================================
eff_demo <- scored %>%
  pivot_longer(all_of(DEMOS), names_to = "demo", values_to = "group") %>% filter(!is.na(group))

# Significance test: do the groups within a demographic differ in optimality?
# Test ONE value per user (mean optimality) - traces from the same user aren't
# independent, so a per-trace test would be pseudoreplicated. Restrict to
# completed_all so the round set is held fixed across users; otherwise a group
# that quits after the easy early rounds averages toward 1.0 and the difference
# is round-mix, not skill (the Q10 confound). Optimality is bounded and
# left-skewed, so use Kruskal-Wallis (non-parametric) rather than ANOVA.
# eps2 = H/(n-1) is the effect size: with thousands of users almost any gap is
# "significant", so eps2 (small ~0.01, medium ~0.06, large ~0.14) carries the
# magnitude that the p-value alone does not.
q4_user <- scored %>%
  filter(completed_all) %>%
  group_by(user_id) %>%
  summarise(mean_opt = mean(optimality), across(all_of(DEMOS), first), .groups = "drop")

q4_tests <- map_dfr(DEMOS, function(d) {
  dd <- q4_user %>% transmute(group = .data[[d]], mean_opt) %>% filter(!is.na(group))
  kt <- kruskal.test(mean_opt ~ group, data = dd)
  tibble(demo = d, n = nrow(dd), k = n_distinct(dd$group),
         H = unname(kt$statistic), df = unname(kt$parameter),
         p = kt$p.value, eps2 = unname(kt$statistic) / (nrow(dd) - 1))
})
message("\nQ4 Kruskal-Wallis on per-user mean optimality (completed_all pool):")
print(q4_tests %>% mutate(H = round(H, 2), eps2 = round(eps2, 4), p = signif(p, 3)))

q4_lab <- q4_tests %>%
  mutate(lab = sprintf("Kruskal-Wallis  p %s\n\u03b5\u00b2 = %.3f  (n = %s users)",
                       if_else(p < 1e-3, "< 0.001", sprintf("= %.3f", p)),
                       eps2, scales::comma(n)))

save_fig(
  ggplot(eff_demo, aes(group, optimality)) +
    geom_boxplot(fill = "#5b8cff", alpha = 0.5, outlier.alpha = 0.08, linewidth = 0.4) +
    geom_label(data = q4_lab, aes(x = -Inf, y = -Inf, label = lab), inherit.aes = FALSE,
               hjust = 0, vjust = 0, family = "mono", size = 2.6, color = "grey20",
               fill = alpha("white", 0.7), label.size = 0, lineheight = 0.95) +
    facet_wrap(~demo, scales = "free_x", ncol = 2) +
    coord_cartesian(ylim = c(0.6, 1)) +
    labs(title = "Optimality by demographic group",
         subtitle = "optimality = optimal moves / player moves (1.0 = the Concorde optimum). Boxes show the per-trace distribution.\nKruskal-Wallis tests one value per user (mean optimality over completed-all players, so the round set is fixed);\n\u03b5\u00b2 is the effect size (H/(n-1)). With thousands of users tiny gaps reach significance - read \u03b5\u00b2 for the real magnitude.",
         x = NULL, y = "optimality") +
    theme_mow + theme(axis.text.x = element_text(angle = 25, hjust = 1, size = 8)),
  "q4a_optimality_by_demographic.png", 11, 8)

# ===========================================================================
# Q4c. One regression with ALL demographics at once.
#      The Kruskal-Wallis tests above are univariate (one demographic at a
#      time), so a real effect in one can leak into another whenever they're
#      correlated (e.g. younger players game more - is the "gaming" gap just
#      age?). A single linear model of per-user mean optimality on all five
#      demographics estimates each effect while holding the others fixed, and
#      its coefficients read off directly as optimality-point gaps vs a chosen
#      reference level - a simpler, all-in-one summary than five separate tests.
#      Same response as the test above: one value per user, completed_all pool
#      (independent observations, round set fixed). OLS on the per-user mean is
#      fine for interpretation at this n; coefficients are differences in
#      optimality, 95% CI from the t distribution.
# ===========================================================================
q4_reg <- q4_user %>%
  drop_na(all_of(DEMOS)) %>%                       # complete cases for one joint model
  mutate(age          = relevel(factor(age),          ref = "18-29"),
         style        = relevel(factor(style),        ref = "I just start and figure it out"),
         gaming       = relevel(factor(gaming),       ref = "Rarely or never"),
         hand         = relevel(factor(hand),         ref = "Right"),
         optimization = relevel(factor(optimization), ref = "Rarely or never"))

fit <- lm(mean_opt ~ age + style + gaming + hand + optimization, data = q4_reg)
sm  <- summary(fit)
message(sprintf("\nQ4c regression: mean optimality ~ all demographics (complete-case n = %d, R^2 = %.3f, adj = %.3f)",
                nrow(q4_reg), sm$r.squared, sm$adj.r.squared))
print(round(sm$coefficients, 4))
message("\nQ4c term significance (drop1 F-test: each demographic net of all the others):")
print(drop1(fit, test = "F"))

ci <- confint(fit)
coef_tbl <- tibble(term     = rownames(sm$coefficients),
                   estimate = sm$coefficients[, "Estimate"],
                   se       = sm$coefficients[, "Std. Error"],
                   p        = sm$coefficients[, "Pr(>|t|)"],
                   lo       = ci[, 1], hi = ci[, 2]) %>%
  filter(term != "(Intercept)") %>%
  mutate(demo  = DEMOS[map_int(term, ~ which(startsWith(.x, DEMOS))[1])],
         level = str_remove(term, paste0("^", demo)),
         sig   = p < 0.05)

# reference rows (gap = 0 by definition) so each facet shows its baseline
ref_tbl <- tibble(demo  = DEMOS,
                  level = c("18-29", "I just start and figure it out",
                            "Rarely or never", "Right", "Rarely or never"),
                  estimate = 0)
# natural within-demographic ordering (ordinal where it applies)
ord <- unique(c("Under 18", "18-29", "30-44", "45-59", "60+",
                "I just start and figure it out", "I have a rough approach but adapt as I go",
                "I follow a set pattern", "I've never done either",
                "Rarely or never", "Sometimes", "Regularly",
                "Right", "Left", "Ambidextrous"))
plot_tbl <- bind_rows(mutate(coef_tbl, ref = FALSE), mutate(ref_tbl, ref = TRUE)) %>%
  mutate(level = factor(level, levels = rev(ord)),
         demo  = factor(demo, levels = DEMOS))
xr <- max(abs(c(coef_tbl$lo, coef_tbl$hi)), na.rm = TRUE)

save_fig(
  ggplot(plot_tbl, aes(estimate, level)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey55") +
    geom_pointrange(data = filter(plot_tbl, !ref),
                    aes(xmin = lo, xmax = hi, color = sig), fatten = 2.2) +
    geom_point(data = filter(plot_tbl, ref), shape = 21, fill = "white",
               color = "grey45", size = 2.4, stroke = 0.6) +
    facet_grid(demo ~ ., scales = "free_y", space = "free_y", switch = "y") +
    scale_color_manual(values = c(`TRUE` = "#ff5c8a", `FALSE` = "grey60"),
                       labels = c(`TRUE` = "p < 0.05", `FALSE` = "not significant"),
                       name = NULL, na.translate = FALSE) +
    coord_cartesian(xlim = c(-xr, xr)) +
    labs(title = sprintf("Optimality regression: all demographics together (R\u00b2 = %.3f, adj %.3f)",
                         sm$r.squared, sm$adj.r.squared),
         subtitle = sprintf("OLS of per-user mean optimality on all five demographics at once (completed-all pool, complete-case n = %s).\nPoint = that level's gap vs its reference (hollow point at 0); bars = 95%% CI; pink = p < 0.05.\nEvery effect is net of the others - unlike the one-at-a-time Kruskal-Wallis tests.",
                            scales::comma(nrow(q4_reg))),
         x = "optimality difference vs reference level", y = NULL) +
    theme_mow + theme(strip.placement = "outside",
                      strip.text.y.left = element_text(angle = 0, face = "bold", size = 8),
                      panel.spacing = unit(4, "pt"),
                      legend.position = "top"),
  "q4c_optimality_regression.png", 12, 8)

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

# Q6 (granular): the average path split by solver cohort - do the best/worst and
# fastest/slowest players sweep the lawn differently? Rows = round, cols = cohort.
save_fig(
  ggplot(filter(cells_co, trace_share >= 0.5), aes(x, y, fill = mean_step_frac)) +
    geom_tile() +
    facet_wrap(~ level + cohort, scales = "free", ncol = 4) +
    scale_y_reverse() +
    scale_fill_viridis_c(option = "plasma", labels = scales::percent,
                         name = "avg point in\npath (0=start,\n1=end)") +
    labs(title = "The average path by solver cohort",
         subtitle = "Rows = round, columns = cohort (top/worst by optimality, quick/slow by time; each ~10% of the completed-all pool).\nColour = when, on average, that cohort reaches each cell. White gaps = obstacles.",
         x = NULL, y = NULL) +
    theme_mow + theme_grid + theme(strip.text = element_text(face = "bold", size = 8)),
  "q6d_visit_heatmaps_by_cohort.png", 13, 18)

# most common exact path per round, derived from the raw point sequences
sigs <- pts %>%
  arrange(trace_id, i) %>%
  group_by(trace_id) %>%
  summarise(level = first(level),
            sig = paste(x, y, sep = ",", collapse = ";"), .groups = "drop")
modal <- sigs %>%
  count(level, sig, name = "cnt") %>%
  group_by(level) %>%
  summarise(n_traces = sum(cnt), modal_count = max(cnt), distinct_paths = n(),
            modal_sig = sig[which.max(cnt)], .groups = "drop") %>%
  mutate(modal_share = round(modal_count / n_traces, 4),
         level = factor(level, levels = LEVELS, ordered = TRUE))
parse_sig <- function(s) {
  xy <- str_split_fixed(str_split(s, ";")[[1]], ",", 2)
  tibble(x = as.integer(xy[, 1]), y = as.integer(xy[, 2]))
}
labs_df <- modal %>%
  mutate(lab = sprintf("%s\n%.0f%% of traces (%d of %d) · %d distinct routes",
                       level, 100 * modal_share, modal_count, n_traces, distinct_paths))
paths <- modal %>%
  mutate(path = map(modal_sig, parse_sig)) %>%
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
opt <- optimal_raw %>%
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

# Q7 (granular): pause-rate heatmap split by solver cohort - do the slow/worst
# solvers hesitate across more of the lawn than the quick/top solvers?
save_fig(
  ggplot(filter(cells_co, is.finite(pause_rate)), aes(x, y, fill = pause_rate)) +
    geom_tile() +
    facet_wrap(~ level + cohort, scales = "free", ncol = 4) +
    scale_y_reverse() +
    scale_fill_viridis_c(option = "mako", direction = -1, labels = scales::percent,
                         limits = c(0, 0.3), oob = scales::squish,
                         name = "chance a visit\ntriggers a pause\n(capped at 30%)") +
    labs(title = "Where each solver cohort stops to think",
         subtitle = "Rows = round, columns = cohort. Per cell: share of visits followed by a >=1s pause (capped at 30%; the start corner is off-scale).",
         x = NULL, y = NULL) +
    theme_mow + theme_grid + theme(strip.text = element_text(face = "bold", size = 8)),
  "q7b_pause_heatmaps_by_cohort.png", 13, 18)

# ===========================================================================
# Q8. Thinking vs execution, by solver cohort.
#     Each solver cohort (top/worst by optimality, quick/slow by time; each
#     ~10% of the completed-all pool) vs an "all players" baseline column.
#     Cohorts overlap (a user can be both top and slow), so each is built as
#     its own trace subset rather than a single mutually-exclusive grouping.
#     Rows = share of MOVES / share of TIME; columns = baseline + each cohort.
# ===========================================================================
think_exec_shares <- function(df, grp_label) {
  df %>%
    group_by(level) %>%
    summarise(`moves: thinking`  = sum(thinking_moves) / sum(moves),
              `moves: execution` = 1 - sum(thinking_moves) / sum(moves),
              `time: thinking`   = sum(thinking_ms) / sum(thinking_ms + execution_ms),
              `time: execution`  = sum(execution_ms) / sum(thinking_ms + execution_ms),
              .groups = "drop") %>%
    mutate(grp = grp_label)
}

split <- bind_rows(
  think_exec_shares(scored, "all players"),
  map_dfr(seq_along(SOLVER_COHORTS), function(j)
    think_exec_shares(filter(scored, .data[[SOLVER_COHORTS[j]]] %in% TRUE),
                      SOLVER_LABELS[j]))
) %>%
  pivot_longer(c(-level, -grp), names_to = "kv", values_to = "share") %>%
  separate(kv, c("facet", "kind"), sep = ": ") %>%
  mutate(kind = factor(kind, levels = c("execution", "thinking")),
         facet = recode(facet, moves = "share of MOVES", time = "share of TIME"),
         grp = factor(grp, levels = c("all players", SOLVER_LABELS)))

save_fig(
  ggplot(split, aes(level, share, fill = kind)) +
    geom_col(width = 0.8) +
    geom_text(aes(label = ifelse(share > 0.06, scales::percent(share, 1), "")),
              position = position_stack(vjust = 0.5), size = 2.4, color = "white") +
    facet_grid(facet ~ grp) +
    scale_fill_manual(values = c(execution = "#5b8cff", thinking = "#f5b841")) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "Thinking vs execution, by solver cohort",
         subtitle = "Columns = all players, then each solver cohort (top/worst by optimality, quick/slow by time; each ~10% of the completed-all pool).\nA \"thinking\" move is one preceded by a wait of >=1 second.\nShare of MOVES = thinking vs fluent moves; share of TIME = total time spent in each.",
         x = NULL, y = NULL, fill = NULL) +
    theme_mow + theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 7),
                      strip.text.y = element_text(angle = -90)),
  "q8_think_vs_exec.png", 14, 7.5)

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
# Q13. Cohort map: every completed-all player placed by their optimality and
#      time percentiles (ranked within the pool). Dashed lines mark the COHORT_Q
#      tails that define the solver cohorts; a player can sit in an optimality
#      tail (top/worst, vertical bands) and a speed tail (quick/slow, horizontal
#      bands) at once - the corners are those overlaps.
# ===========================================================================
q13 <- user_stats %>%
  filter(completed_all, is.finite(opt_pct), is.finite(dur_pct)) %>%
  mutate(cohort = factor(case_when(top_solver   ~ "top",
                                   worst_solver ~ "worst",
                                   quick_solver ~ "quick",
                                   slow_solver  ~ "slow",
                                   TRUE         ~ "core"),
                         levels = c("top", "worst", "quick", "slow", "core")))
q13_rho <- cor(q13$dur_pct, q13$opt_pct, method = "spearman")

save_fig(
  ggplot(q13, aes(dur_pct, opt_pct)) +
    geom_hline(yintercept = c(COHORT_Q, 1 - COHORT_Q), linetype = "dashed", color = "grey75") +
    geom_vline(xintercept = c(COHORT_Q, 1 - COHORT_Q), linetype = "dashed", color = "grey75") +
    geom_point(aes(color = cohort), alpha = 0.55, size = 1.4) +
    geom_smooth(method = "lm", se = FALSE, color = "grey20", linewidth = 0.7) +
    scale_color_manual(values = c(top = "#3ddc97", worst = "#ff5c8a",
                                  quick = "#5b8cff", slow = "#f5b841", core = "grey80")) +
    scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(labels = scales::percent) +
    coord_equal() +
    labs(title = "Cohort map: optimality vs time percentile (completed-all players)",
         subtitle = sprintf("One dot per player, ranked within the %s who finished every round.\nDashed lines = the %d%% cohort tails (top/worst by optimality, quick/slow by time).\nSlower players score modestly higher (Spearman rho = %.2f) \u2014 the top tier deliberates.",
                            scales::comma(nrow(q13)), round(COHORT_Q * 100), q13_rho),
         x = "duration percentile (slower \u2192)", y = "optimality percentile (better \u2191)",
         color = "cohort") +
    theme_mow,
  "q13_cohort_percentile_map.png", 11, 8)

# ===========================================================================
# Q14. Distribution of per-move wait times (ALL moves)
#   One observation per move = dt, the wait *before* it (t[i]-t[i-1]). This is
#   the raw quantity behind the trace-level thinking_* and cell-level pause_*
#   summaries. Waits span ms to multi-minute idle pauses, so plot on a log10
#   axis; the dashed line is PAUSE_MS, the 1s threshold that splits "thinking"
#   from fluent "execution".
# ===========================================================================
waits <- pts %>%
  filter(!is.na(dt)) %>%
  transmute(trace_id, level = factor(level, levels = LEVELS, ordered = TRUE),
            dt_ms = dt, is_pause = dt >= PAUSE_MS)
# log axis needs dt > 0 (a few moves share a timestamp -> dt = 0)
wait_pos <- waits %>% filter(dt_ms > 0)
message(sprintf("Q14: %d moves total, %d with dt>0 (%d zero-wait dropped from log plot)",
                nrow(waits), nrow(wait_pos), nrow(waits) - nrow(wait_pos)))

save_fig(
  ggplot(wait_pos, aes(dt_ms)) +
    geom_histogram(aes(fill = dt_ms >= PAUSE_MS), bins = 60,
                   color = "white", linewidth = 0.1) +
    geom_vline(xintercept = PAUSE_MS, linetype = "dashed", color = "grey20") +
    annotate("text", x = PAUSE_MS, y = Inf, label = "  1s thinking threshold",
             hjust = 0, vjust = 1.6, family = "mono", size = 3.4, color = "grey25") +
    scale_x_log10(labels = scales::comma) +
    scale_fill_manual(values = c(`FALSE` = "#5b8cff", `TRUE` = "#f5b841"),
                      labels = c("execution (<1s)", "thinking (>=1s)"), name = NULL) +
    labs(title = "Distribution of per-move wait times (all moves)",
         subtitle = sprintf("One observation per move = the wait before it (dt = t[i]-t[i-1]). %s moves across %s traces.\nLog time axis; dashed line marks the 1s threshold splitting fluent execution from deliberate thinking.",
                            scales::comma(nrow(wait_pos)), scales::comma(n_distinct(waits$trace_id))),
         x = "wait before move (ms, log scale)", y = "moves") +
    theme_mow,
  "q14_move_wait_histogram.png", 10, 6)

# Same distribution split per round (waits get longer / heavier-tailed on big grids)
save_fig(
  ggplot(wait_pos, aes(dt_ms)) +
    geom_histogram(aes(fill = level), bins = 50, color = "white",
                   linewidth = 0.1, show.legend = FALSE) +
    geom_vline(xintercept = PAUSE_MS, linetype = "dashed", color = "grey20") +
    facet_wrap(~level, scales = "free_y", ncol = 2) +
    scale_x_log10(labels = scales::comma) +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Per-move wait-time distribution by round",
         subtitle = "Dashed line = 1s thinking threshold. Log time axis; y free per round.",
         x = "wait before move (ms, log scale)", y = "moves") +
    theme_mow,
  "q14b_move_wait_histogram_by_round.png", 10, 7)

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
