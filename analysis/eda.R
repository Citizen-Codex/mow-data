library(tidyverse)
library(jsonlite)


raw <- read.csv("data/mow_test_rows.csv")

optimal_moves <- c(
  "0" = 40,
  "1" = 40,
  "2" = 82,
  "3" = 170
)

optimal_moves[[as.character(0)]]

processed <- raw %>%
  select(-created_at) %>%
  rename(moves = result) %>%
  mutate(
    moves = map(moves, fromJSON),
    config = factor(config)
  )

augmented <- processed %>%
  mutate(
    first_move_time = map_dbl(moves, \(x) x$t[2]),
    first_five_move_time = map_dbl(moves, \(x) x$t[6]),
    total_time = map_dbl(moves, \(x) x$t[length(x$t)]),
    total_moves = map_dbl(moves, \(x) length(x$t)),
    moves = map(moves, \(x) mutate(x, dt = t - lag(t))),
    optimal_moves = map_dbl(config, \(x) optimal_moves[[x]]),
    extra_moves = total_moves - optimal_moves
  )

augmented %>%
  select(user, config, moves) %>%
  mutate(id = paste(user, config, sep = "_")) %>%
  unnest(moves) %>%
  group_by(id) %>%
  mutate(step = row_number()) %>%
  ungroup() %>%
  ggplot(aes(x = step, y = dt, group = user, color = user)) +
  geom_line(alpha = 0.6, na.rm = TRUE) +
  geom_hline(yintercept = 1000, linetype = "dashed") +
  facet_wrap(~config, ncol = 1, scales = "free_y") +
  labs(
    title = "Move Delta Time by Run",
    x = "Move Number",
    y = "Wait Time (ms)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

augmented %>%
  select(user, config, moves) %>%
  mutate(id = paste(user, config, sep = "_")) %>%
  unnest(moves) %>%
  group_by(id) %>%
  mutate(step = row_number()) %>%
  group_by(config, step) %>%
  summarise(dt = mean(dt)) %>%
  ungroup() %>%
  ggplot(aes(x = step, y = dt)) +
  geom_line(alpha = 0.6, na.rm = TRUE) +
  geom_hline(yintercept = 1000, linetype = "dashed") +
  facet_wrap(~config, ncol = 1, scales = "free_y") +
  labs(
    title = "Move Delta Time by Run",
    x = "Move Number",
    y = "Wait Time (ms)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

augmented %>%
  select(user, config, moves) %>%
  mutate(id = paste(user, config, sep = "_")) %>%
  unnest(moves) %>%
  ggplot(aes(x = dt, group = user, fill = user)) +
  geom_histogram(alpha = 1, na.rm = TRUE) +
  facet_wrap(~config, ncol = 1) +
  labs(
    title = "Move Delta Time Histogram by Run",
    x = "Wait Time (ms)",
    y = "Frequency"
  ) +
  scale_x_log10() +
  theme_minimal() +
  geom_vline(xintercept = 1000, linetype = "dashed") +
  theme(legend.position = "none")


augmented %>%
  ggplot(aes(x = first_move_time, y = extra_moves)) +
  geom_point(alpha = 0.6, na.rm = TRUE) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Extra Moves vs. First Move Time",
    x = "First Move Time (ms)",
    y = "Extra Moves"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


model <- lm(extra_moves ~ total_time + first_move_time, data = augmented)

summary(model)
plot(model)
