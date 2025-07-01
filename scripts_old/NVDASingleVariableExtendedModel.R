library(tidyverse)
library(keras3)
library(zoo)

# Read the data
nvda_df <- read.csv("nvda_stock_data.csv") %>%
  mutate(Weekday = as.factor(Weekday), 
         DayNum = as.numeric((DayNum - min(DayNum)) / (max(DayNum) - min(DayNum))), 
         DaysToEarnings = as.numeric((DaysToEarnings - min(DaysToEarnings)) / (max(DaysToEarnings) - min(DaysToEarnings))), 
         Month = as.numeric((Month - min(Month)) / (max(Month) - min(Month))), 
         DaysToEarnings = as.numeric((DaysToEarnings - min(DaysToEarnings)) / (max(DaysToEarnings) - min(DaysToEarnings)))) %>%
         select(-Date)

nvda_df$Weekday <- as.numeric(factor(nvda_df$Weekday, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")))

# Convert Weekday to numeric, where Monday = 1, Friday = 5
nvda_df$Weekday <- as.numeric((nvda_df$Weekday - min(nvda_df$Weekday)) / (max(nvda_df$Weekday) - min(nvda_df$Weekday)))

create_sliding_window <- function(data, window_size, prediction_days, target_col) {
  n <- nrow(data)
  features <- ncol(data)
  
  # Initialize X and Y
  X <- array(NA, dim = c(n - window_size - prediction_days + 1, window_size, features))
  Y <- matrix(NA, nrow = n - window_size - prediction_days + 1, ncol = prediction_days)
  
  for (i in seq_len(n - window_size - prediction_days + 1)) {
    # Extract the current window
    current_window <- as.matrix(data[i:(i + window_size - 1), ])
    
    # Calculate mean and SD for standardization (column-wise)
    window_means <- colMeans(current_window, na.rm = TRUE)
    window_sds <- apply(current_window, 2, sd, na.rm = TRUE)
    
    # Avoid division by zero during standardization
    window_sds[window_sds == 0] <- 1
    
    # Standardize the window
    X[i, , ] <- sweep(current_window, 2, window_means, FUN = "-")
    X[i, , ] <- sweep(X[i, , ], 2, window_sds, FUN = "/")
    
    # Standardize the target values (closing prices for the next `prediction_days` days)
    for (j in 1:prediction_days) {
      target_value <- data[[target_col]][i + window_size + j - 1]  # Corrected access of target column
      Y[i, j] <- (target_value - window_means[target_col]) / window_sds[target_col]
    }
  }
  
  return(list(X = X, Y = Y))
}

# Define the parameters
window_size <- 5
prediction_days <- 5
target_col <- "Close"  # The column name for the target (e.g., "Close")

# Generate standardized X and Y
result <- create_sliding_window(nvda_df, window_size, prediction_days, target_col)

# Assign X and Y
X <- result$X
Y <- result$Y

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
  #layer_conv_1d(filters = 64, kernel_size = 3, activation = 'linear') %>%
  layer_lstm(units = 80, input_shape = c(window_size, n_features), return_sequences = FALSE) %>%
  #layer_batch_normalization() %>% 
  #layer_dropout(rate = .6) %>%
  #layer_lstm(units = 30, input_shape = c(window_size, n_features), return_sequences = FALSE) %>%
  layer_dropout(rate = .3) %>%
  layer_dense(units = 5, activation = 'linear')

# Compile the model
model %>%
  compile(optimizer = optimizer_adam(learning_rate = 0.0001), 
          loss = 'mean_squared_error')

# Summarize the model
summary(model)

# Train the model
history <- model %>%
  fit(trainX, trainY, epochs = 600, batch_size = 128, validation_split = 0.2)

# Evaluate the model on the test set
test_loss <- model %>% evaluate(testX, testY)
test_loss[[1]]

# Get the last window from testX (most recent data)
last_window <- testX[dim(testX)[1]-2, , ]


# Predict the next 5 days using the model
predicted_values <- predict(model, tail(testX))

predicted_values




