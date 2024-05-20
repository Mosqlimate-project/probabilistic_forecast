# probabilistic_forecast
 The repository contains the code that generates the data used on the probabilistic forecast map of the platform.

## Model structure
The model is based on three LSTM layers concatenated. It uses the data from the last 8 weeks to predict the next 4 weeks. 

| Layer (type) | Output Shape | Param # |
| --- | --- | --- |
| input_layer (InputLayer) | (1, 8, 561) | 0 |
| lstm (LSTM) | (1, 8, 16) | 36,992 |
| dropout (Dropout) | (1, 8, 16) | 0 |
| lstm_1 (LSTM) | (1, 8, 16) | 2,112 |
| dropout_1 (Dropout) | (1, 8, 16) | 0 |
| lstm_2 (LSTM) | (1, 16) | 2,112 |
| dropout_2 (Dropout) | (1, 16) | 0 |
| dense (Dense) | (1, 4) | 68 |
 Total params: 41,284 (161.27 KB)
 Trainable params: 41,284 (161.27 KB)

## Preparing the Environment
In order to run this code, you need to configure a virtual environment with the required packages. You can use [poetry](https://python-poetry.org/) for this. Before proceding make sure you have poetry installed on your machine. If you don't have it, you can [install it](https://python-poetry.org/docs/#installation) by running the following command:
```bash
 $ pip install poetry
 ```
After installing poetry, you can create the virtual environment by running the following commands:
```bash
 $ poetry install
 $ poetry shell
```

## Training the models
 The model was trained by the script `lstm/train_models_to_for.py`. 

## Forecasting

 The forecast is done with the script `lstm/apply_forecast.py`. 
