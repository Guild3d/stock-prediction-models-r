library(tidyverse)
library(keras3)
library(zoo)

# Read the data
nvda_df <- read.csv("nvda_stock_data.csv") %>%
  mutate(
    Weekday = as.factor(Weekday), 
    DayNum = as.numeric((DayNum - min(DayNum)) / (max(DayNum) - min(DayNum))), 
    DaysToEarnings = as.numeric((DaysToEarnings - min(DaysToEarnings)) / (max(DaysToEarnings) - min(DaysToEarnings))), 
    Month = as.numeric((Month - min(Month)) / (max(Month) - min(Month))), 
    DaysToEarnings = as.numeric((DaysToEarnings - min(DaysToEarnings)) / (max(DaysToEarnings) - min(DaysToEarnings))), 
    Open = log(Open), Close = log(Close), High = log(High), Low = log(Close)) %>%
  select(-Date)

nvda_df$Weekday <- as.numeric(factor(nvda_df$Weekday, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")))

# Convert Weekday to numeric, where Monday = 1, Friday = 5
nvda_df$Weekday <- as.numeric((nvda_df$Weekday - min(nvda_df$Weekday)) / (max(nvda_df$Weekday) - min(nvda_df$Weekday)))

create_sliding_window <- function(data, window_size, prediction_days, threshold = 0.015) {
  n <- nrow(data)
  features <- ncol(data)
  X <- array(NA, dim = c(n - window_size - prediction_days + 1, window_size, features))
  Y <- array(NA, dim = c(n - window_size - prediction_days + 1, prediction_days))
  
  for (i in seq_len(n - window_size - prediction_days + 1)) {
    # Create input window
    X[i, , ] <- as.matrix(data[i:(i + window_size - 1), ])
    
    # Compute percentage change and classify into up, down, or flat
    future_close <- data$Close[(i + window_size):(i + window_size + prediction_days - 1)]
    current_close <- data$Close[i + window_size - 1]
    percentage_change <- (future_close - current_close) / current_close
    
    Y[i, ] <- ifelse(
      abs(percentage_change) < threshold,  # Flat
      0, 
      ifelse(percentage_change > 0, 1, 2)  # Up or Down
    )
  }
  
  return(list(X = X, Y = Y))
}

# Set the window size and prediction days (e.g., predict 5 days)
window_size <- 5
prediction_days <- 5
threshold <- 0.015

# Generate sliding windows (X and Y)
data_windows <- create_sliding_window(nvda_df, window_size, prediction_days, threshold)
X <- data_windows$X
Y <- data_windows$Y

# Verify dimensions
dim(X)  # Should be (samples, window_size, n_features)
dim(Y)  # Should be (samples, prediction_days)

# One-hot encode Y for classification
Y_onehot <- array(0, dim = c(dim(Y)[1], prediction_days, 3))
for (i in seq_len(prediction_days)) {
  Y_onehot[, i, ] <- to_categorical(Y[, i], num_classes = 3)
}

# Training/Test splits (unchanged)
trainX <- X[1:(dim(X)[1] - 35), , ]
trainY <- Y_onehot[1:(dim(Y_onehot)[1] - 35), , ]
testX <- X[(dim(X)[1] - 34):dim(X)[1], , ]
testY <- Y_onehot[(dim(Y_onehot)[1] - 34):dim(Y_onehot)[1], , ]

# Create the model
n_features <- ncol(nvda_df)

set.seed(1)
model <- keras_model_sequential() %>%
  #layer_conv_1d(filters = 64, kernel_size = 3, activation = 'relu', input_shape = c(window_size, n_features)) %>%
  layer_lstm(units = 90, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = prediction_days * 3, activation = 'softmax') %>%
  layer_reshape(target_shape = c(prediction_days, 3))  # Reshape to (days, classes)

# Compile the model
model %>%
  compile(
    optimizer = optimizer_adam(learning_rate = 0.0001), 
    loss = 'categorical_crossentropy', 
    metrics = c('accuracy')
  )

# Summarize the model
summary(model)

# Train the model
history <- model %>%
  fit(trainX, trainY, epochs = 300, batch_size = 64, validation_split = 0.2)

# Evaluate the model on the test set
test_loss <- model %>% evaluate(testX, testY)
cat("Test Loss: ", test_loss[[1]], "\n")

# Get the last window from testX (most recent data)
last_window <- testX[dim(testX)[1], , ]

# Predict the next 5 days using the model
predicted_probs <- predict(model, tail(testX))
predicted_classes <- apply(predicted_probs, c(1, 2), which.max) - 1  # Convert one-hot to classes
print(predicted_classes)
