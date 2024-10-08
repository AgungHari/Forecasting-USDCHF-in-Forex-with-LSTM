
# Forecasting USDCHF in Forex with LSTM


![IPyKernel version](https://img.shields.io/badge/IPyKernel-v6.29.4-yellow)
![Pandas version](https://img.shields.io/badge/pandas-v2.2.3-black)
![Pytorch version](https://img.shields.io/badge/pytorch-v2.4.1+cu118-red) 
![ScikitLearn version](https://img.shields.io/badge/scikitlearn-v1.5.2-blue)

Forecasting USDCHF in Forex with LSTM is a machine learning-based project aimed at predicting the price of the USD/CHF currency pair in Forex using an LSTM (Long Short-Term Memory) model. This project uses historical exchange rate data of USD/CHF, particularly the closing price (Close), to train a model capable of predicting future price movements over multiple timesteps.

Disclaimer : This project is not open source


## Demo

<img alt="GitHub License" src="https://img.shields.io/github/license/AgungHari/TinyBERT-Enhanced-Chat-System-for-Mobile-Legends">

![Stock](https://github.com/user-attachments/assets/de639752-ae12-4be5-b441-80fbca829e09)


This test uses 20 timestamps with a 5-minute interval from October 3, 2024, 21:55 to 23:30. Here is the sequence:
```bash
new_data = [0.85192, 0.85256, 0.85242, 0.85234, 0.85232, 0.8523, 0.85229, 0.85228, 0.85229, 0.85227, 
            0.85228, 0.85217, 0.85212, 0.85208, 0.85207, 0.85211, 0.85206, 0.85197, 0.85193, 0.85194]
```

The output is a prediction for 5 steps ahead. here is the output for the current version of the model : AgungHari_Predict_USDCHF_4.h5 

```bash
prediction = [0.8519057 , 0.85185087, 0.8519225 , 0.85189843, 0.852037855]
```

Actual price :

```bash
actual_price = [0.85209, 0.85188, 0.85182, 0.85181, 0.85184]
```



## Background

Forex (Foreign Exchange) is a vast and highly liquid global financial market where currency pairs are traded. USD/CHF is one of the major currency pairs, representing the exchange rate between the US Dollar (USD) and the Swiss Franc (CHF). The price movements of this currency pair are influenced by various factors such as monetary policy, global economic conditions, and changes in interest rates.
## Project Goals

The main goal of this project is to build an LSTM model capable of predicting the price movements of the USD/CHF pair for several steps ahead (multi-step forecasting). This project aims to assist forex traders by:

- Predicting future prices of the USD/CHF currency pair.
- Identifying market trends to make better trading decisions.
- Optimizing trading strategies based on technical analysis using AI predictions.
## Workflow

Data Collection: Historical data of the USD/CHF pair is obtained from forex data sources, including Open, High, Low, and Close prices at specific time intervals.

Data Preprocessing: The obtained data is processed by removing noise, normalizing it with MinMaxScaler, and formatting it into time-series datasets suitable for LSTM model input.

LSTM Model Development: The LSTM model is trained using historical USD/CHF data. The model learns from past price patterns to predict future price movements.

Prediction and Visualization: The trained model is used to predict the future price of the USD/CHF pair for multiple timesteps. The prediction results are visually displayed by comparing actual data and the model's predictions.
## Authors

<img alt="Static Badge" src="https://img.shields.io/badge/AgungHari-black?style=social&logo=github&link=https%3A%2F%2Fgithub.com%2FAgungHari">

<img alt="Static Badge" src="https://img.shields.io/badge/AbelMarcel-black?style=social&logo=github&link=https://github.com/AbelMarcelR">


