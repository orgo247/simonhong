# visualizations.R (clean, runnable)
# Assumes files live in data/ and use Kaggle names + one merged daily CSV from analysis.py

library(dplyr)
library(readr)
library(lubridate)
library(ggplot2)
library(hms)

DATA_DIR <- "data"
FIG_DIR  <- "outputs/figures"
if (!dir.exists(FIG_DIR)) dir.create(FIG_DIR, recursive = TRUE)

# Inputs (adjust here if you rename anything)
daily_merged_path     <- file.path(DATA_DIR, "daily_merged.csv")           # created by analysis.py
hourly_steps_path     <- file.path(DATA_DIR, "hourlySteps_merged.csv")
hourly_intensity_path <- file.path(DATA_DIR, "hourlyIntensities_merged.csv")
hourly_calories_path  <- file.path(DATA_DIR, "hourlyCalories_merged.csv")

# ---- Load data ----
daily <- read_csv(daily_merged_path, show_col_types = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    TotalActiveMinutes = VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes,
    SleepEfficiency = if_else(TotalTimeInBed > 0, 100 * TotalMinutesAsleep / TotalTimeInBed, NA)
  )

hourly_steps <- read_csv(hourly_steps_path, show_col_types = FALSE) %>%
  mutate(ActivityHour = parse_date_time(ActivityHour, orders = c("mdY IMS p","Y-m-d H:M:S")),
         hour_of_day = hour(ActivityHour),
         StepTotal = suppressWarnings(as.numeric(StepTotal)))

hourly_intensity <- read_csv(hourly_intensity_path, show_col_types = FALSE) %>%
  mutate(ActivityHour = parse_date_time(ActivityHour, orders = c("mdY IMS p","Y-m-d H:M:S")),
         hour_of_day = hour(ActivityHour))

hourly_calories <- read_csv(hourly_calories_path, show_col_types = FALSE) %>%
  mutate(ActivityHour = parse_date_time(ActivityHour, orders = c("mdY IMS p","Y-m-d H:M:S")),
         hour_of_day = hour(ActivityHour))

# ---- 1) Activity level distribution (daily) ----
activity_minutes <- daily %>%
  summarize(
    avg_very_active   = mean(VeryActiveMinutes, na.rm = TRUE),
    avg_fairly_active = mean(FairlyActiveMinutes, na.rm = TRUE),
    avg_lightly       = mean(LightlyActiveMinutes, na.rm = TRUE),
    avg_sedentary     = mean(SedentaryMinutes, na.rm = TRUE)
  ) %>%
  tidyr::pivot_longer(everything(),
                      names_to = "activity_level",
                      values_to = "avg_minutes") %>%
  mutate(activity_level = factor(activity_level,
                                 levels = c("avg_very_active","avg_fairly_active","avg_lightly","avg_sedentary"),
                                 labels = c("Very Active","Fairly Active","Lightly Active","Sedentary")))

p1 <- ggplot(activity_minutes, aes(activity_level, avg_minutes, fill = activity_level)) +
  geom_col() +
  labs(title = "Average Minutes by Activity Level", x = "Activity Level", y = "Average Minutes") +
  theme_minimal() + theme(legend.position = "none")
ggsave(file.path(FIG_DIR, "activity_minutes.png"), p1, width = 7, height = 4, dpi = 120)

# ---- 2) Steps vs Sleep (daily scatter) ----
p2 <- ggplot(daily, aes(x = TotalMinutesAsleep, y = TotalSteps)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Steps vs. Sleep Time", x = "Total Minutes Asleep", y = "Total Steps") +
  theme_minimal()
ggsave(file.path(FIG_DIR, "steps_vs_sleep.png"), p2, width = 7, height = 4, dpi = 120)

# ---- 3) Hourly trends (steps & intensity) ----
hourly_steps_summary <- hourly_steps %>%
  group_by(hour_of_day) %>%
  summarize(avg_steps = mean(StepTotal, na.rm = TRUE), .groups = "drop")

p3 <- ggplot(hourly_steps_summary, aes(hour_of_day, avg_steps)) +
  geom_line() + geom_point() +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Average Steps per Hour (All Users)", x = "Hour of Day", y = "Average Steps") +
  theme_minimal()
ggsave(file.path(FIG_DIR, "hourly_steps.png"), p3, width = 7, height = 4, dpi = 120)

hourly_intensity_summary <- hourly_intensity %>%
  group_by(hour_of_day) %>%
  summarize(avg_intensity = mean(AverageIntensity, na.rm = TRUE), .groups = "drop")

p4 <- ggplot(hourly_intensity_summary, aes(hour_of_day, avg_intensity)) +
  geom_line() + geom_point() +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Average Hourly Intensity vs Time of Day", x = "Hour of Day", y = "Average Intensity") +
  theme_minimal()
ggsave(file.path(FIG_DIR, "hourly_intensity.png"), p4, width = 7, height = 4, dpi = 120)

# ---- 4) Weekly trends (steps & sleep) ----
daily_week <- daily %>%
  mutate(day_of_week = wday(Date, label = TRUE, abbr = FALSE)) %>%
  group_by(day_of_week) %>%
  summarize(avg_steps = mean(TotalSteps, na.rm = TRUE),
            avg_sleep = mean(TotalMinutesAsleep, na.rm = TRUE), .groups = "drop")

p5 <- ggplot(daily_week, aes(day_of_week, avg_steps)) +
  geom_col() +
  labs(title = "Average Steps by Day of Week", x = "Day of Week", y = "Average Steps") +
  theme_minimal()
ggsave(file.path(FIG_DIR, "weekly_steps.png"), p5, width = 7, height = 4, dpi = 120)

p6 <- ggplot(daily_week, aes(day_of_week, avg_sleep)) +
  geom_col() +
  labs(title = "Average Sleep by Day of Week", x = "Day of Week", y = "Average Sleep (minutes)") +
  theme_minimal()
ggsave(file.path(FIG_DIR, "weekly_sleep.png"), p6, width = 7, height = 4, dpi = 120)