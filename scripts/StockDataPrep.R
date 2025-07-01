library(tidyverse)
library(quantmod)
library(zoo)
library(TTR)

# Set the stock symbol and date range
stock_symbols <- c("GLD", "IWM", "SOXL", "TLT", "XLF")

exportStocks <- function(stock_symbol){
  start_date <- as.Date("6/5/2000", format = "%m/%d/%Y")
  start_cutoff_date <- as.Date("1/01/1990", format = "%m/%d/%Y")
  end_date <- Sys.Date()+1
  
  # Download the stock data
  data <- getSymbols(stock_symbol, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
  
  # Convert to a data frame
  df <- data.frame(Date = index(data), coredata(data)) %>% 
    rename(Open = paste0(stock_symbol,".Open"), High = paste0(stock_symbol,".High"), Low = paste0(stock_symbol,".Low"), Close = paste0(stock_symbol,".Close"), 
           Volume = paste0(stock_symbol,".Volume")) %>% select(-paste0(stock_symbol, ".Adjusted")) %>% 
    mutate(Weekday = weekdays(Date), DayNum = format(Date, "%d"), Month = format(Date, "%m"))
  
  #MACD
  macd_values <- MACD(df$Close, wilder = FALSE, fast = 12, slow = 26, signal = 9)
  RSI_values <- RSI(df$Close, n=14)/100
  df <- bind_cols(df, macd_values, tibble(RSI_Scaled = RSI_values)) %>% rename(MACD = macd, MACDSignal = signal) %>% mutate(MACDCalc = MACD-MACDSignal)
  
  # 50 and 200 day moving averages
  df$SMA50 <- SMA(df$Close, n = 50)
  df$SMA200 <- SMA(df$Close, n = 200)
  
  rolling_window_50 <- 50
  rolling_window_200 <- 200
  
  df <- df %>%
    mutate(
      # Rolling mean and standard deviation for SMA50
      SMA50_RollingMean = rollapply(SMA50, rolling_window_50, mean, partial = TRUE, align = "right"),
      SMA50_RollingSD = rollapply(SMA50, rolling_window_50, sd, partial = TRUE, align = "right"),
      SMA50_Standardized = (SMA50 - SMA50_RollingMean) / SMA50_RollingSD,
      
      # Rolling mean and standard deviation for SMA200
      SMA200_RollingMean = rollapply(SMA200, rolling_window_200, mean, partial = TRUE, align = "right"),
      SMA200_RollingSD = rollapply(SMA200, rolling_window_200, sd, partial = TRUE, align = "right"),
      SMA200_Standardized = (SMA200 - SMA200_RollingMean) / SMA200_RollingSD,
      
      Return_1D = (Close / lag(Close, 1)) - 1,
      Return_3D = (Close / lag(Close, 3)) - 1,
      Return_5D = (Close / lag(Close, 5)) - 1,
      HiLo = High / Low,
      OpenClose = Open / Close
    )
  
  
  # Remove rows with NA from rolling calculations
  df <- df %>% select(-SMA50, -SMA200, -SMA50_RollingMean, -SMA50_RollingSD, -SMA200_RollingMean, -SMA200_RollingSD)
  
  
  df <- df %>%filter(Date >= start_cutoff_date) %>%   mutate(Weekday = as.factor(Weekday), DayNum = as.factor(DayNum), Month = as.factor(Month), Volume = (Volume/100000000))
  
  df <- df %>% mutate(MACD_Crossover = as.integer(MACD > MACDSignal & lag(MACD) <= lag(MACDSignal)),
                      RSI_Overbought = as.integer(RSI_Scaled > 0.7), 
                      RSI_Oversold = as.integer(RSI_Scaled < 0.3),
                      Close_above_SMA200 = as.integer(Close > SMA200_Standardized),
                      Trend_Gradient = (SMA50_Standardized - SMA200_Standardized) / SMA200_Standardized,
                      Close_SMA_Ratio = Close / SMA50_Standardized,
                      MACD_Diff = MACD - MACDSignal)
  
  atr_vals <- ATR(df[, c("High", "Low", "Close")], n = 14)[,2]
  df$ATR <- atr_vals
  
  df <- df %>%drop_na() 
  
  # Export the data to a CSV file
  output_file <- paste0(tolower(stock_symbol), "_stock_data.csv")
  write.csv(df, file = output_file, row.names = FALSE)
  
  # Print message
  cat("Stock data has been saved to", output_file, "\n")
}

for (i in stock_symbols) {
  exportStocks(i)
}
