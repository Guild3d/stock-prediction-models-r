library(tidyverse)
library(keras3)

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

create_sliding_window <- function(data, window_size) {
  n <- nrow(data)
  features <- ncol(data)
  output <- array(NA, dim = c(n - window_size, window_size, features))
  
  for (i in seq_len(n - window_size)) {
    output[i, , ] <- as.matrix(data[i:(i + window_size - 1), ])
  }
  return(output)
}

# Define window size (e.g., 5 days)
window_size <- 5

# Generate sliding windows
X <- create_sliding_window(nvda_df, window_size)
Y <- nvda_df$Close[(window_size+1):(nrow(X)+window_size)]

# Verify the dimensions of the output
dim(X)
length(Y)
tail(Y)

#Training Test Splits
trainX <- X[1:(dim(X)[1]-25),,]
trainY <- Y[1:(length(Y)-25)]
dim(trainX)
length(trainY)

testX <- X[(dim(X)[1]-24):dim(X)[1],,]
testY <- Y[(length(Y)-24):length(Y)]
dim(testX)
length(testY)

#Creating the Model
n_features <- ncol(nvda_df)

set.seed(1)
model <- keras_model_sequential() %>%
  layer_lstm(units = 80, input_shape = c(window_size, n_features), return_sequences = FALSE) %>%
  layer_dropout(rate = 0.25) %>%  # Dropout layer with 20% probability
  layer_dense(units = 1, activation = 'linear')

# Compile the model
model %>%
  compile(optimizer = optimizer_adam(learning_rate = 0.0001), 
          loss = 'mean_squared_error')

# Summarize the model
summary(model)

# Train the model
history <- model %>%
  fit(trainX, trainY, epochs = 1000, batch_size = 50, validation_split = 0.2)

# Evaluate the model on the test set
#test_loss <- model %>% evaluate(testX, testY)
#cat("Test Loss:", test_loss)

#predict(model,testX[25,])

####################

# model <- keras_model_sequential() %>%
#   layer_dense(units = 10, activation = 'relu', input_shape = c(window_size * n_features)) %>%
#   layer_dense(units = 1)  # Single output unit for closing price prediction
# 
# # Compile the model
# model %>%
#   compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = 'mean_squared_error')
# 
# # Flatten input for dense layer
# X_flat <- array_reshape(X, dim = c(nrow(X), window_size * n_features))
# 
# # Train the model
# history <- model %>%
#   fit(X_flat, Y, epochs = 50, batch_size = 32, validation_split = 0.2, verbose = 2)
# 
# 
# 

