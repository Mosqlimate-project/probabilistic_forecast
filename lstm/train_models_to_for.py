import pandas as pd
from train_models import train_dl_model
from lstm import build_model, build_lstm  #custom_loss_msle

dfs = pd.read_csv('../data/macro_saude.csv')

PREDICT_N = 4  # number of new days predicted
LOOK_BACK = 8 # number of last days used to make the prediction
BATCH_SIZE = 1
EPOCHS = 300
#HIDDEN = 
L1 = 1e-6
L2 = 1e-6
LOSS = 'mean_squared_error'

#for macro in dfs.code_macro.unique():
#for macro in dfs.code_macro.unique():
for macro in ['3103']:
#for macro in dfs.state.unique():

    if macro in ['AC', 'AL', 'AP', 'RN', 'RO', 'RR', 'SE', 'TO']:
        HIDDEN = 32
    else:
        HIDDEN = 16

    if (str(macro)[:2] == '41') or (str(macro)[:2] == '42') or (str(macro)[:2] == '43'):

        TRAIN_FROM = '2021-06-01'
        HIDDEN = 16
    else: 
        TRAIN_FROM = '2015-06-01'

    print(f'Training Macroregion: {macro}')

    FILENAME_DATA = f'../data/dengue_{macro}.csv.gz'

    end_date = '2024-04-21'
    # if os.path.exists(f'../saved_models/lstm/trained_{macro}_dengue_macro.keras'):
    #     continue
    df_ = pd.read_csv(FILENAME_DATA, index_col='Unnamed: 0', nrows = 1)

    feat = df_.shape[1]

    model = build_lstm(hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
                        batch_size=BATCH_SIZE, loss=LOSS, 
                optimizer = 'adam' )
    
    #build_model(l1=L1, l2=L2, hidden=HIDDEN, features=feat, predict_n=PREDICT_N, look_back=LOOK_BACK,
            #            batch_size=BATCH_SIZE, loss=LOSS, lr=0.001)

    train_dl_model(model, macro, doenca='dengue',
                   end_date_train=None,
                   ratio=1,
                   ini_date=TRAIN_FROM,
                   end_date=end_date,
                   plot=False, filename_data=FILENAME_DATA,
                   min_delta=0.001, label='macro_4',
                   patience = 30, 
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   predict_n=PREDICT_N,
                   look_back=LOOK_BACK)
