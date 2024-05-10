# probabilistic_forecast
 The repository contains the code that generates the data used on the probabilistic forecast map of the platform.

 The model was trained on the script `lstm/train_models_to_for.py`. The model is based on three LSTM layers concatenated. It uses the data from the last 8 weeks to predict the next 4 weeks. 

 The forecast is done on the script `lstm/apply_forecast.py`. 
