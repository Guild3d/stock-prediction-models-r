# ETF Stock Prediction in R

This project explores short-term directional prediction of ETF prices (e.g., GLD, IWM) using machine learning in R. Models include an LSTM built in Keras and a parallelized XGBoost pipeline. The project focuses on improving predictive accuracy using confidence filtering and leakage-aware preprocessing.

---

## Project Highlights

- **LSTM Model (Keras)**: Multi-day price forecasting with rolling standardization  
- **XGBoost Pipeline**: Parallelized backtesting across multiple ETFs and thresholds  
- **Confidence Filtering**: Improves accuracy by only acting on high-confidence signals  

---

## Tools Used

- R, Python, Keras, xgboost  
- quantmod, tidyverse, doParallel, foreach

---

## Results

- ~80% accuracy on confidence filtered predictions 
- ~12% coverage on actionable signals  
- Fast parallel model evaluation using XGBoost

---

## Visuals

Example output and variable performance graphs included in the model output folder

---

## Author

**Gabriel Love**  
[LinkedIn](https://www.linkedin.com/in/gabriel-love-17a39b209/)
