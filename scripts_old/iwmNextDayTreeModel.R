library(randomForest)
library(dplyr)

# Assuming you have a dataframe `df` with RSI, MACD, SMA, and Date features
# and 'Close' column as the closing prices

df <- read.csv("iwm_stock_data.csv")

df <- df %>%
  mutate(
    next_close = lead(Close, 1),  # Get the next day's closing price
    pct_change = (next_close - Close) / Close * 100,  # Calculate percentage change
    target = case_when(
      pct_change > .01 ~ 'up',      # Percentage change > 0.75% for 'up'
      pct_change < -.01 ~ 'down',   # Percentage change < -0.75% for 'down'
      TRUE ~ NA               # Else, it's neutral
    )
  ) %>%
  drop_na()  # Remove rows with NA values (due to lead)

# Convert target to a factor
df$target <- factor(df$target, levels = c('down', 'up'))

df <- df %>% select(-Date, -pct_change, -next_close)
df <- df %>% select(-Weekday, -DayNum, -Month)

# Split data into training and test sets
set.seed(1)
train_index <- sample(1:nrow(df), size = 0.8 * nrow(df))
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Train the random forest model
rf_model <- randomForest(target ~ ., data = train_data, importance = TRUE, ntree=1200)

# View model summary
print(rf_model)
importance(rf_model)

# Predict on test data
predictions <- predict(rf_model, test_data, type = "response")
prediction_probs <- predict(rf_model, test_data, type = "prob")


# Add predictions and probabilities to the test data
test_data$predicted_class <- predictions
test_data$prob_up <- prediction_probs[, "up"]
test_data$prob_down <- prediction_probs[, "down"]

test_data$max_prob <- pmax(test_data$prob_up, test_data$prob_down)
conf_threshold <- 0.7

confident_preds <- test_data %>% filter(max_prob >= conf_threshold)

# 2. Evaluate accuracy on confident predictions
accuracy <- mean(confident_preds$predicted_class == confident_preds$target)
cat("Confidence-filtered accuracy:", round(accuracy * 100, 2), "%\n")

# 3. Confusion matrix
cat("Confusion matrix (filtered):\n")
print(table(predicted = confident_preds$predicted_class, actual = confident_preds$target))

# 4. Volume of filtered predictions
cat("Fraction of test set kept:", nrow(confident_preds), "/", nrow(test_data), 
    "(", round(nrow(confident_preds)/nrow(test_data) * 100, 2), "% )\n")