import sys
import tensorflow.keras as keras
import pandas as pd

sys.path.append('../')
from plots_lstm import plot_loss
from lstm import build_model, train_model

PREDICT_N = 10  # number of new days predicted
LOOK_BACK = 12  # number of last days used to make the prediction
BATCH_SIZE = 1
EPOCHS = 300
HIDDEN = 16
L1 = 1e-5
L2 = 1e-5
TRAIN_FROM = '2015-06-01'  # Train the models from this date
LOSS = 'msle'

def train_dl_model(model, city, doenca='dengue', end_date_train='2022-11-01', ratio=None, end_date='2023-12-31',
                   ini_date=TRAIN_FROM,
                   plot=True, filename_data=f'../data/dengue.csv', patience=20, min_delta=0.001,
                   label=LOSS,
                   look_back=LOOK_BACK, predict_n=PREDICT_N, batch_size=BATCH_SIZE,
                   epochs=EPOCHS, verbose = 0):

    #model = build_model(l1=l1, l2=l2, hidden=hidden, features=feat, predict_n=predict_n, look_back=look_back,
     #                   batch_size=batch_size, loss=loss, lr=lr)

    model, hist = train_model(model, city, doenca=doenca, epochs=epochs, end_train_date=end_date_train,
                              ini_date=ini_date,
                              ratio=ratio, end_date=end_date,
                              predict_n=predict_n, look_back=look_back, label=label,
                              batch_size=batch_size,
                              filename=filename_data, verbose=verbose, patience=patience, min_delta=min_delta)

    if plot:
        plot_loss(hist, title=F'Model loss - {city}')
