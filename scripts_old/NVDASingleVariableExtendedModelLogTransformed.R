library(tidyverse)
library(keras3)
library(zoo)

# Read the data
nvda_df <- read.csv("nvda_stock_data.csv") %>%
  mutate(Weekday = as.factor(Weekday), 
         DayNum = as.numeric((DayNum - min(DayNum)) / (max(DayNum) - min(DayNum))), 
         DaysToEarnings = as.numeric((DaysToEarnings - min(DaysToEarnings)) / (max(DaysToEarnings) - min(DaysToEarnings))), 
         Month = as.numeric((Month - min(Month)) / (max(Month) - min(Month))), 
         DaysToEarnings = as.numeric((DaysToEarnings - min(DaysToEarnings)) / (max(DaysToEarnings) - min(DaysToEarnings))), 
         Open = log(Open), Close = log(Close), High = log(High), Low = log(Close)) %>%
         select(-Date)

nvda_df$Weekday <- as.numeric(factor(nvda_df$Weekday, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")))

# Convert Weekday to numeric, where Monday = 1, Friday = 5
nvda_df$Weekday <- as.numeric((nvda_df$Weekday - min(nvda_df$Weekday)) / (max(nvda_df$Weekday) - min(nvda_df$Weekday)))

create_sliding_window <- function(data, window_size, prediction_days) {
  n <- nrow(data)
  features <- ncol(data)
  output <- array(NA, dim = c(n - window_size - prediction_days + 1, window_size, features))
  
  for (i in seq_len(n - window_size - prediction_days + 1)) {
    output[i, , ] <- as.matrix(data[i:(i + window_size - 1), ])
  }
  
  return(output)
}

# Set the window size and prediction days (e.g., predict 5 days)
window_size <- 5
prediction_days <- 5

# Generate sliding windows (X)
X <- create_sliding_window(nvda_df, window_size, prediction_days)
Y <- nvda_df$Close[(window_size + 1):(nrow(nvda_df) - prediction_days + 1)]
Y <- cbind(Y, nvda_df$Close[(window_size + 2):(nrow(nvda_df) - prediction_days + 2)],
           nvda_df$Close[(window_size + 3):(nrow(nvda_df) - prediction_days + 3)],
           nvda_df$Close[(window_size + 4):(nrow(nvda_df) - prediction_days + 4)],
           nvda_df$Close[(window_size + 5):(nrow(nvda_df) - prediction_days + 5)])

# Verify dimensions
dim(X)  # Should be (samples, window_size, n_features)
dim(Y)  # Should be (samples, 5)



#Training Test Splits
trainX <- X[1:(dim(X)[1]-35),,]
trainY <- Y[1:(dim(Y)[1]-35),]
dim(trainX)
dim(trainY)

testX <- X[(dim(X)[1]-34):dim(X)[1],,]
testY <- Y[(dim(Y)[1]-34):dim(Y)[1],]
dim(testX)
dim(testY)

#Creating the Model
n_features <- ncol(nvda_df)

set.seed(1)
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 64, kernel_size = 3, activation = 'linear') %>%
  layer_lstm(units = 90, input_shape = c(window_size, n_features), return_sequences = FALSE) %>%
  #layer_batch_normalization() %>% 
  #layer_dense(units = 60, activation = 'linear', kernel_regularizer = regularizer_l2(l = .01)) %>%
  #layer_dropout(rate = .7) %>%
  #layer_lstm(units = 30, input_shape = c(window_size, n_features), return_sequences = FALSE) %>%
  layer_dropout(rate = .2) %>%
  layer_dense(units = 5, activation = 'linear')

# Compile the model
model %>%
  compile(optimizer = optimizer_adam(learning_rate = 0.0001), 
          loss = 'mean_absolute_error')

# Summarize the model
summary(model) 

# Train the model
history <- model %>%
  fit(trainX, trainY, epochs = 3350, batch_size = 128, validation_split = 0.2)

 # Evaluate the model on the test set
test_loss <- model %>% evaluate(testX, testY)
exp(test_loss[[1]])

# Get the last window from testX (most recent data)
last_window <- testX[dim(testX)[1], , ]

# Predict the next 5 days using the model
predicted_values <- predict(model, tail(testX))

exp(predicted_values)*exp(test_loss[[1]])
dim(trainX[1,,])


dim(tail(testX))
