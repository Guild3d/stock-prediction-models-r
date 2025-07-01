library(dplyr)
library(xgboost)

# Load data
df <- read.csv("iwm_stock_data.csv")

df <- df %>%
  mutate(
    Return_1D = (Close / lag(Close, 1)) - 1,
    Return_3D = (Close / lag(Close, 3)) - 1,
    Return_5D = (Close / lag(Close, 5)) - 1
  )

df <- df %>%
  mutate(
    next_close = lead(Close, 1),
    pct_change = (next_close - Close) / Close * 100,
    target = case_when(
      pct_change > 0.6 ~ 'up',
      pct_change < -0.6 ~ 'down',
      TRUE ~ NA_character_
    )
  ) %>%
  drop_na()  # Drop NA rows from lead or class assignment

# Drop non-feature columns
df <- df %>% mutate(HiLo = High/Low, OpenClose = Open/Close)
df <- df %>% select(-Date, -pct_change, -next_close, -Weekday, -DayNum, -Month, -Low, -High, -Return_5D, -OpenClose)

# Encode target as numeric for xgboost (1 = up, 0 = down)
df$target_numeric <- ifelse(df$target == "up", 1, 0)

# Prepare train/test split
set.seed(1)
train_index <- sample(1:nrow(df), size = 0.8 * nrow(df))
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Set up feature columns (exclude target)
feature_cols <- setdiff(colnames(df), c("target", "target_numeric"))

# Convert to matrix format for xgboost
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, feature_cols]), label = train_data$target_numeric)
dtest  <- xgb.DMatrix(data = as.matrix(test_data[, feature_cols]), label = test_data$target_numeric)

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
test_data$prob_up <- prob_up
test_data$prob_down <- 1 - prob_up
test_data$predicted_class <- ifelse(prob_up > 0.5, "up", "down")

# Evaluate performance
test_data$target <- factor(test_data$target, levels = c("down", "up"))
test_data$predicted_class <- factor(test_data$predicted_class, levels = c("down", "up"))

conf_matrix <- table(predicted = test_data$predicted_class, actual = test_data$target)
print(conf_matrix)

# -----------------------------------------------
# CONFIDENCE FILTERING
# -----------------------------------------------

# Set confidence threshold (you can adjust this)
threshold <- 0.7

# Filter rows where the model is confident in its prediction
test_data$max_prob <- pmax(test_data$prob_up, test_data$prob_down)
confident_preds <- test_data %>% filter(max_prob >= threshold)

# Evaluate accuracy on confident predictions only
confident_accuracy <- mean(confident_preds$predicted_class == confident_preds$target)

cat("Confidence-filtered accuracy (>", threshold, "): ",
    round(confident_accuracy * 100, 2), "%\n", sep = "")

# Confusion matrix for confident predictions
cat("Confusion matrix (confident predictions only):\n")
print(table(predicted = confident_preds$predicted_class,
            actual = confident_preds$target))

# Coverage (what % of total test set is kept)
coverage <- nrow(confident_preds) / nrow(test_data)
cat("Coverage: ", round(coverage * 100, 2), "% of test set\n", sep = "")

importance_matrix <- xgb.importance(model = xgb_model)

# View top features
print(importance_matrix)

