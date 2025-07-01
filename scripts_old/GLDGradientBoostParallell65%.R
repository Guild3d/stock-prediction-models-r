library(tidyverse)
library(xgboost)
library(foreach)
library(future)
library(doFuture)
library(progressr)
library(doRNG)

# Load data
df <- read.csv("gld_stock_data.csv")
df$Weekday <- as.integer(factor(df$Weekday, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")))


# Parameters
lookback_window <- 1500  # days to train on before each prediction
horizon <- 5             # how far into the future to predict
threshold <- 0.8       # confidence threshold
up <- 0
down <- 0
down <- -down
features <- c( "MACD", "MACDSignal", "RSI_Scaled", "SMA200_Standardized", "MACDCalc","Return_1D", "Return_3D","SMA50_Standardized", "Return_5D", "Volume" )

#Available Features
#"Open", "High", "Low", "Close", "Volume", "Weekday", "DayNum", "Month", "MACD", "MACDSignal", "RSI_Scaled", 
#"MACDCalc", "SMA50_Standardized", "SMA200_Standardized","Return_1D", "Return_3D", "Return_5D", "HiLo", "OpenClose"

# Preallocate results
results <- data.frame()

#Parallel Processing
plan(multisession, workers = parallel::detectCores() - 1)
registerDoFuture()
handlers(global = TRUE)

total_iterations <- nrow(df) - horizon - lookback_window

# Loop through each day for rolling prediction
start_i <- nrow(df) - horizon - 250
end_i <- nrow(df) - horizon

suppressWarnings(
  results <- with_progress({
    p <- progressor(steps = length(start_i:end_i))
    
    foreach(i = (start_i):(end_i),
            .combine = rbind, 
            .packages = c("dplyr", "xgboost", "tidyr", "forcats")) %dorng% {
              p()
              
              set.seed(i)
              # Build training set (up to day i-1)
              df_train <- df[(i - lookback_window):(i - 1), ] %>%
                mutate(
                  next_close = lead(Close, horizon),
                  pct_change = (next_close - Close) / Close * 100,
                  target = case_when(
                    pct_change > up ~ 'up',
                    pct_change < down ~ 'down',
                    TRUE ~ NA_character_
                  ),
                  Return_1D = (Close / lag(Close, 1)) - 1,
                  Return_3D = (Close / lag(Close, 3)) - 1,
                  Return_5D = (Close / lag(Close, 5)) - 1,
                  HiLo = High / Low,
                  OpenClose = Open / Close,
                  target_numeric = ifelse(target == "up", 1, 0)
                ) %>%
                mutate(target = ifelse(is.na(target), sample(c("up", "down"), 
                                                             size = sum(is.na(target)), replace = TRUE), target),
                       target_numeric = ifelse(target == "up", 1, 0)) %>%
                select(-Date, -pct_change, -next_close)
              
              if (nrow(df_train) < 50) return(NULL)
              
              feature_cols <- setdiff(features, c("target", "target_numeric"))
              
              weights = ifelse(df_train$target_numeric == 0, 7, 1)
              dtrain <- xgb.DMatrix(data = as.matrix(df_train[, feature_cols]), label = df_train$target_numeric, weight = weights)
              
              # Train model
              model <- xgb.train(
                params = list(
                  objective = "binary:logistic",
                  eval_metric = c("logloss", "auc", "error"),
                  eta = 0.02,
                  max_depth = 7,
                  subsample = 0.8,
                  colsample_bytree = 0.8,
                  lambda = 2,
                  alpha = 1,
                  scale_pos_weight = 12
                ),
                data = dtrain,
                nrounds = 800,
                verbose = 0
              )
              
              # Build prediction features for day i
              if (i < 6) return(NULL)
              row <- df[i, ] %>%
                mutate(
                  Return_1D = (Close / df$Close[i - 1]) - 1,
                  Return_3D = (Close / df$Close[i - 3]) - 1,
                  Return_5D = (Close / df$Close[i - 5]) - 1,
                  HiLo = High / Low,
                  OpenClose = Open / Close
                )
              row_input <- row %>% select(all_of(feature_cols))
              dtest <- xgb.DMatrix(data = as.matrix(row_input))
              
              # Predict
              prob_up <- predict(model, dtest)
              prob_down <- 1 - prob_up
              pred_class <- ifelse(prob_up > 0.5, "up", "down")
              max_prob <- pmax(prob_up, prob_down)
              
              actual_future <- df$Close[i + horizon]
              actual_class <- case_when(
                (actual_future - df$Close[i]) / df$Close[i] * 100 > up ~ "up",
                (actual_future - df$Close[i]) / df$Close[i] * 100 < down ~ "down",
                TRUE ~ "neutral"
              )
              
              data.frame(
                Date = df$Date[i],
                Prediction = pred_class,
                Confidence = max_prob,
                Actual = actual_class
              )
            }
  })
)
# Filter by confidence
confident_results <- results %>% filter(Confidence >= threshold, !is.na(Actual))

# Evaluate
accuracy <- mean(confident_results$Prediction == confident_results$Actual)
coverage <- nrow(confident_results) / nrow(results)

results %>%
  mutate(correct = Prediction == Actual) %>%
  group_by(cut(Confidence, breaks = seq(0.5, 1, 0.1))) %>%
  summarise(accuracy = mean(correct), n = n())


cat("\nConfidence-filtered Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Coverage:", round(coverage * 100, 2), "% of all predictions\n")

# Confusion matrix
print(table(predicted = confident_results$Prediction, actual = confident_results$Actual))
feature_cols <- setdiff(features, c("target", "target_numeric"))
df_train %>%
  pivot_longer(cols = all_of(feature_cols)) %>%
  ggplot(aes(x = value, fill = target)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~name, scales = "free")
