library(tidyverse)
library(randomForest)
library(zoo)

# Read and preprocess the data
nvda_df <- read.csv("nvda_stock_data.csv") %>%
  mutate(
    Weekday = as.factor(Weekday), 
    DayNum = as.numeric((DayNum - min(DayNum)) / (max(DayNum) - min(DayNum))), 
    DaysToEarnings = as.numeric((DaysToEarnings - min(DaysToEarnings)) / (max(DaysToEarnings) - min(DaysToEarnings))), 
    Month = as.numeric((Month - min(Month)) / (max(Month) - min(Month))), 
    Open = log(Open), Close = log(Close), High = log(High), Low = log(Low)
  ) %>%
  select(-Date)

nvda_df$Weekday <- as.numeric(factor(nvda_df$Weekday, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")))

# Sliding window function
create_rf_data_recursive <- function(data, window_size, threshold = 0.015) {
  n <- nrow(data)
  features <- ncol(data)
  rf_data <- data.frame()
  
  for (i in seq_len(n - window_size - 1)) {
    # Flatten the sliding window into a single row
    window <- as.vector(t(data[i:(i + window_size - 1), ]))
    
    # Calculate the target: percentage change for the next day
    future_close <- data$Close[i + window_size]
    current_close <- data$Close[i + window_size - 1]
    percentage_change <- (future_close - current_close) / current_close
    
    # Classify into up, down, or flat
    target <- ifelse(abs(percentage_change) < threshold, 0, ifelse(percentage_change > 0, 1, 2))
    
    rf_data <- rbind(rf_data, c(window, target))
  }
  
  colnames(rf_data) <- c(paste0("Feature_", 1:(window_size * features)), "Target")
  return(rf_data)
}

# Set window size and threshold
window_size <- 5
threshold <- 0.015

# Generate training data
rf_data <- create_rf_data_recursive(nvda_df, window_size, threshold)
rf_data$Target <- as.factor(rf_data$Target)

train_rf <- rf_data[1:(nrow(rf_data) - 35), ]
test_rf <- rf_data[(nrow(rf_data) - 34):nrow(rf_data), ]

# Train the random forest model
set.seed(1)
rf_model <- randomForest(
  Target ~ ., 
  data = train_rf, 
  ntree = 500, 
  mtry = sqrt(ncol(train_rf) - 1), 
  importance = TRUE
)

# Predict recursively for the next 5 days
recursive_predictions <- matrix(NA, nrow = nrow(test_rf), ncol = 5)

for (i in seq_len(5)) {
  if (i == 1) {
    # Initial predictions
    recursive_predictions[, i] <- predict(rf_model, newdata = test_rf)
  } else {
    # Update features with previous predictions
    for (j in 1:nrow(test_rf)) {
      test_rf[j, paste0("Feature_", (window_size * ncol(nvda_df)) - (i - 1))] <- 
        recursive_predictions[j, i - 1]
    }
    recursive_predictions[, i] <- predict(rf_model, newdata = test_rf)
  }
}

# Evaluate performance
table(Predicted = recursive_predictions[, 1], Actual = test_rf$Target)

# Inspect feature importance
importance(rf_model)
