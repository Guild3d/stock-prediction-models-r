library(tidyverse)
library(xgboost)

# Load data
df <- read.csv("tlt_stock_data.csv")

# Step 1: Chronological split
n <- nrow(df)
split_index <- nrow(df)-120

df_train_raw <- df[1:split_index, ]
df_test_raw  <- df[(split_index + 1):n, ]

# Step 2: Add target to train set
df_train <- df_train_raw %>%
  mutate(
    next_close = lead(Close, 5),
    pct_change = (next_close - Close) / Close * 100,
    target = case_when(
      pct_change > 0.3 ~ 'up',
      pct_change < -0.3 ~ 'down',
      TRUE ~ NA_character_
    ),
    Return_1D = (Close / lag(Close, 1)) - 1,
    Return_3D = (Close / lag(Close, 3)) - 1,
    Return_5D = (Close / lag(Close, 5)) - 1
  ) %>%
  drop_na() %>%
  mutate(
    HiLo = High / Low,
    OpenClose = Open / Close,
    target_numeric = ifelse(target == "up", 1, 0)
  ) %>%
  select(-Date, -pct_change, -next_close, -Weekday, -DayNum, -Month, -Low, -High, -SMA200_Standardized, -SMA50_Standardized, )

# Step 3: Add target to test set
df_test <- df_test_raw %>%
  mutate(
    next_close = lead(Close, 5),
    pct_change = (next_close - Close) / Close * 100,
    target = case_when(
      pct_change > 0.3 ~ 'up',
      pct_change < -0.3 ~ 'down',
      TRUE ~ NA_character_
    ),
    Return_1D = (Close / lag(Close, 1)) - 1,
    Return_3D = (Close / lag(Close, 3)) - 1,
    Return_5D = (Close / lag(Close, 5)) - 1
  ) %>%
  drop_na() %>%
  mutate(
    HiLo = High / Low,
    OpenClose = Open / Close,
    target_numeric = ifelse(target == "up", 1, 0)
  ) %>%
  select(-Date, -pct_change, -next_close, -Weekday, -DayNum, -Month, -Low, -High, -SMA200_Standardized, -SMA50_Standardized, )

# Step 4: Set feature columns and create matrices
feature_cols <- setdiff(colnames(df_train), c("target", "target_numeric"))

dtrain <- xgb.DMatrix(data = as.matrix(df_train[, feature_cols]), label = df_train$target_numeric)
dtest  <- xgb.DMatrix(data = as.matrix(df_test[, feature_cols]), label = df_test$target_numeric)
set.seed(1)
# Train xgboost model
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 400,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# Predict probabilities and class
prob_up <- predict(xgb_model, newdata = dtest)
df_test$prob_up <- prob_up
df_test$prob_down <- 1 - prob_up
df_test$predicted_class <- ifelse(prob_up > 0.5, "up", "down")

# Evaluate performance
df_test$target <- factor(df_test$target, levels = c("down", "up"))
df_test$predicted_class <- factor(df_test$predicted_class, levels = c("down", "up"))

conf_matrix <- table(predicted = df_test$predicted_class, actual = df_test$target)
print(conf_matrix)

# -----------------------------------------------
# CONFIDENCE FILTERING
# -----------------------------------------------
# Set confidence threshold (you can adjust this)
threshold <- 0.7

# Filter rows where the model is confident in its prediction
df_test$max_prob <- pmax(df_test$prob_up, df_test$prob_down)
confident_preds <- df_test %>% filter(max_prob >= threshold)

# Evaluate accuracy on confident predictions only
confident_accuracy <- mean(confident_preds$predicted_class == confident_preds$target)

cat("Confidence-filtered accuracy (>", threshold, "): ",
    round(confident_accuracy * 100, 2), "%\n", sep = "")

# Confusion matrix for confident predictions
cat("Confusion matrix (confident predictions only):\n")
print(table(predicted = confident_preds$predicted_class,
            actual = confident_preds$target))

# Coverage (what % of total test set is kept)
coverage <- nrow(confident_preds) / nrow(df_test_raw)
cat("Coverage: ", round(coverage * 100, 2), "% of test set\n", sep = "")

# Variable importance
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)

# Identify errors
confident_preds <- confident_preds %>%
  mutate(correct = predicted_class == target)

errors <- confident_preds %>% filter(!correct)
correct_preds <- confident_preds %>% filter(correct)

# -----------------------------------------------
# Today's Predictions
# -----------------------------------------------
# Get 5 most recent rows with Close
recent_rows <- df %>%
  tail(5) %>%
  mutate(
    HiLo = High / Low,
    OpenClose = Open / Close,
    Return_1D = (Close / lag(Close, 1)) - 1,
    Return_3D = (Close / lag(Close, 3)) - 1,
    Return_5D = (Close / lag(Close, 5)) - 1
  )

recent_features <- recent_rows %>%
  select(all_of(feature_cols))

# Convert to DMatrix
recent_matrix <- xgb.DMatrix(as.matrix(recent_features))

# Predict probabilities
recent_prob_up <- predict(xgb_model, recent_matrix)
recent_confidence <- pmax(recent_prob_up, 1 - recent_prob_up)
recent_class <- ifelse(recent_prob_up > 0.5, "up", "down")
recent_close <- recent_rows$Close

confidence_pass <- ifelse(recent_confidence >= threshold, "✅", "❌")

# Create output data frame
# Define the index range of the last 5 rows
last_rows <- (nrow(df) - 4):nrow(df)

# Format dates as "Apr 24, 2025"
formatted_dates <- format(as.Date(df$Date[last_rows]), "%b %d, %Y")

# Build the results table
results <- data.frame(
  Date = formatted_dates,
  Close = round(recent_close, 2),
  Prediction = recent_class,
  Confidence = round(recent_confidence * 100, 2),
  Pass = confidence_pass
)

# Print the table
print(results, right = FALSE, row.names = FALSE)
cat("Confidence-filtered accuracy (>", threshold, "): ", round(confident_accuracy * 100, 2), "%\n", sep = "")
cat("Coverage: ", round(coverage * 100, 2), "% of test set\n", sep = "")
cat("Stock: TLT")
