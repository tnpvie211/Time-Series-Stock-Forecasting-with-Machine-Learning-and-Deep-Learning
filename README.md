# Project Overview
Built an end-to-end time-series forecasting pipeline to predict historical stock prices using Python. Retrieved multi-year stock market data via the Yahoo Finance API, performed data cleaning, normalization, and feature scaling, and engineered training/test datasets for supervised learning.
Implemented and evaluated both a baseline Linear Regression model and an LSTM deep learning model to compare predictive performance on sequential financial data. Visualized actual vs. predicted prices and measured model accuracy using MSE and RMSE, demonstrating how sequence-aware models such as LSTM outperform baseline regression approaches when modeling temporal dependencies in financial data.

## Objectives
- Extract historical stock price data using an external API
- Clean, normalize, and prepare time-series data for modeling
- Build a baseline Linear Regression model for comparison
- Implement an LSTM neural network for stock price prediction
- Evaluate and visualize model performance

## Data Sources
- Yahoo Finance API (via yahooquery)
- Historical stock prices including:
  - Open
  - Close
  - Volume

## Tools & Technologies
- Programming Language: Python
- Data Extraction: Yahoo Finance API (yahooquery)
- Data Processing: pandas, NumPy
- Machine Learning: scikit-learn
- Deep Learning: Keras / TensorFlow
- Visualization: Matplotlib
 
## Project Workflow
1. Data Extraction
- Retrieved historical stock data using the Yahoo Finance API
- Selected a multi-year time range for analysis

2. Data Preprocessing
- Removed irrelevant features
- Normalized numerical variables using Min-Max scaling
- Converted raw data into structured, model-ready datasets
- Saved preprocessed data for reproducibility

3. Baseline Model: Linear Regression
- Built a Linear Regression model to predict closing prices
- Scaled input and target variables for stable training
- Evaluated performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

4. Deep Learning Model: LSTM
- Transformed time-series data into sequential input windows
- Designed an LSTM neural network using Keras
- Trained and validated the model on historical price data
- Compared predictions against actual closing prices

5. Model Evaluation & Visualization
- Visualized actual vs. predicted stock prices
- Compared baseline and LSTM model performance
- Demonstrated improved accuracy using sequence-based modeling

## Results
- Linear Regression provided a baseline prediction performance
- LSTM models captured temporal patterns more effectively
- Sequence-based deep learning demonstrated improved predictive accuracy over traditional regression
