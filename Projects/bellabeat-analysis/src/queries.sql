-- SQL snippets used in the analysis (translated from the PDF). Adjust table/file names to your DB.

-- 1) Merge daily activity and sleep into one table
CREATE TABLE daily_activity_merged AS
SELECT d.Id, d.Date, d.TotalSteps, d.Calories, d.TotalDistance,
       d.VeryActiveMinutes, d.FairlyActiveMinutes, d.LightlyActiveMinutes, d.SedentaryMinutes,
       s.TotalMinutesAsleep, s.TotalTimeInBed, 
       (d.VeryActiveMinutes + d.FairlyActiveMinutes + d.LightlyActiveMinutes) AS TotalActiveMinutes
FROM daily_activity AS d
LEFT JOIN sleep_day AS s
  ON d.Id = s.Id AND d.Date = s.Date;

-- 2) Activity level bucketing
SELECT
  CASE
    WHEN TotalSteps >= 12500 THEN 'High Active'
    WHEN TotalSteps BETWEEN 5000 AND 12499 THEN 'Moderate Active'
    ELSE 'Low Active'
  END AS activity_level,
  COUNT(*) AS total_days_in_level,
  ROUND(AVG(Calories), 0) AS avg_calories,
  ROUND(AVG(TotalMinutesAsleep), 0) AS avg_sleep_minutes
FROM daily_activity_merged
GROUP BY activity_level
ORDER BY avg_calories DESC;

-- 3) Time allocation across activity levels
SELECT
  ROUND(AVG(VeryActiveMinutes), 1)   AS avg_very_active,
  ROUND(AVG(FairlyActiveMinutes), 1) AS avg_fairly_active,
  ROUND(AVG(LightlyActiveMinutes), 1) AS avg_lightly_active,
  ROUND(AVG(SedentaryMinutes), 1)     AS avg_sedentary
FROM daily_activity_merged;

-- 4) Sleep pattern bucketing
SELECT
  CASE WHEN TotalMinutesAsleep >= 420 THEN 'Adequate Sleep' ELSE 'Inadequate Sleep' END AS sleep_pattern,
  COUNT(*) AS total_days_in_pattern,
  ROUND(AVG(TotalSteps), 0) AS avg_steps,
  ROUND(AVG(Calories), 0) AS avg_calories,
  ROUND(AVG(TotalActiveMinutes), 0) AS avg_totalActive
FROM daily_activity_merged
GROUP BY sleep_pattern
ORDER BY total_days_in_pattern DESC;

-- 5) Engagement level per user
SELECT
  Id,
  CASE
    WHEN COUNT(*) >= 60 THEN 'High Engagement'
    WHEN COUNT(*) BETWEEN 45 AND 59 THEN 'Moderate Engagement'
    ELSE 'Low Engagement'
  END AS engagement_level,
  ROUND(AVG(TotalSteps)) AS avg_steps,
  ROUND(AVG(TotalMinutesAsleep)) AS avg_sleep
FROM daily_activity_merged
GROUP BY Id
ORDER BY engagement_level;

-- 6) Hourly pattern (requires hourly tables)
SELECT
  EXTRACT(HOUR FROM hs.Time) AS hour_of_day,
  ROUND(AVG(hs.StepTotal), 0)  AS avg_steps,
  ROUND(AVG(hc.Calories), 0)   AS avg_calories,
  ROUND(AVG(hi.TotalIntensity), 2) AS avg_intensity
FROM hourly_steps hs
JOIN hourly_calories hc ON hs.Id = hc.Id AND hs.Date = hc.Date AND hs.Time = hc.Time
JOIN hourly_intensities hi ON hs.Id = hi.Id AND hs.Date = hi.Date AND hs.Time = hi.Time
GROUP BY EXTRACT(HOUR FROM hs.Time)
ORDER BY hour_of_day;

-- 7) Weekly trends
SELECT
  DAYNAME(Date) AS day_of_week,
  ROUND(AVG(TotalSteps), 0) AS avg_steps,
  ROUND(AVG(TotalMinutesAsleep), 0) AS avg_sleep
FROM daily_activity_merged
GROUP BY DAYNAME(Date)
ORDER BY FIELD(day_of_week, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday');

-- 8) Weight tracking utilization
SELECT COUNT(*) AS total_logs, COUNT(DISTINCT Id) AS unique_weight_users FROM weight_log;