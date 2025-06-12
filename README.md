# S&P 500 Price Prediction Using Enhanced Temporal Convolutional Networks

This repository contains a deep learning project that forecasts the S&P 500 index using **Enhanced Temporal Convolutional Networks (TCN)**. The model integrates multiple data sources and applies innovative forecasting strategies to improve prediction accuracy and robustness.

## Project Overview

This project focuses on predicting the next day's closing price of the **S&P 500 index** using a combination of **financial indicators**, **macroeconomic data**, and **sentiment analysis** from financial news. The model leverages **Enhanced Temporal Convolutional Networks (TCNs)**, a type of deep learning model, to handle the complex time-series nature of the data and capture both long- and short-term dependencies.

Two different prediction strategies were implemented:
1. **Averaging across Sequence Lengths**: Models trained on different sequence lengths are averaged to smooth predictions.
2. **Adaptive Sequence Selection**: A dynamic mechanism selects the best model per time point to improve accuracy during volatile market periods.

## Data Acquisition

The project uses three main data sources:
- **Alpha Vantage API**: Retrieves historical price data and technical indicators such as Moving Averages (SMA), Relative Strength Index (RSI), and Bollinger Bands.
- **FRED API**: Provides macroeconomic data like interest rates (Federal Funds Rate) and the 10-year treasury yield.
- **News API**: Fetches real-time financial headlines for sentiment analysis. Sentiment scores are calculated using **TextBlob**.

These data sources are then preprocessed with techniques like feature scaling, lagging, and wavelet denoising.

## Model Architecture

The **Temporal Convolutional Network (TCN)** used in this project processes time-series data using the following components:
- **Causal Convolutions**: Ensures no future leakage by using past and present data only.
- **Dilated Convolutions**: Increases the receptive field exponentially, allowing the network to capture long-term dependencies.
- **Residual Connections**: Helps in stabilizing training for deep networks.
- **Adaptive Weights**: The multi-branch model dynamically adjusts the contribution of each sequence length at each prediction step.

The model architecture involves stacking several layers of these components to process data with different temporal resolutions.

## Training Methodology

The model was trained using **PyTorch** and **PyTorch Lightning** with the following strategies:
1. **Fixed Sequence Averaging**: Trains multiple models on different sequence lengths and averages their predictions.
2. **Adaptive Sequence Selection**: The model dynamically selects the most accurate prediction at each time step from the different sequence lengths.

We use **Mean Squared Error (MSE)** as the loss function and **Adam Optimizer** for training. The model is trained with early stopping to prevent overfitting.

## Results and Visualization

The results of the model's predictions are visualized in the following figures:

1. **Predictions vs Actuals for Different Sequence Lengths**: Shows how models trained on different sequence lengths (50, 100, 150, 200 timesteps) perform in predicting the S&P 500 index.
2. **Averaged Predictions**: Displays the effectiveness of combining outputs from multiple models.
3. **Adaptive Multi-Scale TCN Forecast**: Demonstrates how the adaptive model selects the best sequence length for each prediction, capturing local reactivity and long-term trends.

## Installation

To run this project, you need to install the following dependencies:

```bash
pip install alpha_vantage pandas numpy yfinance matplotlib tensorflow torch requests textblob pandas-datareader fredapi pywt pytorch-lightning keras-tcn
