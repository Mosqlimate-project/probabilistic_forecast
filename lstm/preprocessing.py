import numpy as np 
import pandas as pd 
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder

def build_lagged_features(dt, lag=2, dropna=True):
    '''
    returns a new DataFrame to facilitate regressing over all lagged features.
    :param dt: Dataframe containing features
    :param lag: maximum lags to compute
    :param dropna: if true the initial rows containing NANs due to lagging will be dropped
    :return: Dataframe
    '''
    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(lag + 1)
        res = pd.concat([dt.shift(-i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res


def split_data(df, look_back=12, ratio=0.8, predict_n=5, Y_column=0):
    """
    Split the data into training and test sets
    Keras expects the input tensor to have a shape of (nb_samples, timesteps, features).
    :param df: Pandas dataframe with the data.
    :param look_back: Number of weeks to look back before predicting
    :param ratio: fraction of total samples to use for training
    :param predict_n: number of weeks to predict
    :param Y_column: Column to predict
    :return:
    """
    df = np.nan_to_num(df.values).astype("float64")
    # n_ts is the number of training samples also number of training sets
    # since windows have an overlap of n-1
    n_ts = df.shape[0] - look_back - predict_n + 1
    # data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    for i in range(n_ts):  # - predict_):
        #         print(i, df[i: look_back+i+predict_n,0])
        data[i, :, :] = df[i: look_back + i + predict_n, :]
    # train_size = int(n_ts * ratio)
    train_size = int(df.shape[0] * ratio) - look_back
    #print(train_size)

    # We are predicting only column 0
    X_train = data[:train_size, :look_back, :]
    Y_train = data[:train_size, look_back:, Y_column]
    X_test = data[train_size:, :look_back, :]
    Y_test = data[train_size:, look_back:, Y_column]

    return X_train, Y_train, X_test, Y_test


def normalize_data(df, log_transform=False, ratio = 0.75, end_train_date = None):
    """
    Normalize features in the example table
    :param df:
    :param ratio: defines the size of the training dataset 
    :return:
    """
    
    if 'municipio_geocodigo' in df.columns:
        df.pop('municipio_geocodigo')

    for col in df.columns:
        if col.startswith('nivel'):
            # print(col)
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])

    df.fillna(0, inplace=True)
    if ratio != None:
        norm, norm_weights = normalize(df.iloc[:int(df.shape[0]*ratio)], norm='max', axis=0, return_norm = True)
        max_values = df.iloc[:int(df.shape[0]*ratio)].max()
    else:
        norm, norm_weights = normalize(df.loc[df.index <= f'{end_train_date}'], norm='max', axis=0, return_norm = True)
        max_values = df.loc[df.index <= f'{end_train_date}'].max()

    df_norm = df.divide(norm_weights, axis='columns')

    if log_transform==True:
        df_norm = np.log(df_norm)

    return df_norm, max_values

def get_nn_data(city, ini_date = None, end_date = None, end_train_date = None, ratio = 0.75, look_back = 4, predict_n = 4, filename = None ):
    """
    :param city: int. The ibge code of the city, it's a seven number code 
    :param ini_date: string or None. Initial date to use when creating the train/test arrays 
    :param end_date: string or None. Last date to use when creating the train/test arrays
    :param end_train_date: string or None. Last day used to create the train data 
    :param ratio: float. If end_train_date is None, we use the ratio to spli the data into train and test 
    :param look_back: int. Number of last days used to make the forecast
    :param predict_n: int. Number of days forecast

    """
    df = pd.read_csv(filename, index_col = 'Unnamed: 0' )
    df.index = pd.to_datetime(df.index)

    try:
        target_col = list(df.columns).index(f"casos_{city}")
    except ValueError:
        target_col = list(df.columns).index(f"casos_est_{city}")

    df = df.dropna()

    if ini_date != None: 
        df = df.loc[ini_date:]

    if end_date != None:
        df = df.loc[:end_date]

    if end_train_date == None: 
        
        norm_df, max_features = normalize_data(df, ratio = ratio)
        factor = max_features[target_col]

        X_train, Y_train, X_test, Y_test = split_data(
                norm_df,
                look_back= look_back,
                ratio=ratio,
                predict_n = predict_n, 
                Y_column=target_col,
        )
    
        # These variables will already concat the train and test array to easy the work of make 
        # the predicions of both 
        X_pred = np.concatenate((X_train, X_test), axis = 0)
        Y_pred = np.concatenate((Y_train, Y_test), axis = 0)

    else:
        norm_df, max_features = normalize_data(df, ratio = None, end_train_date = end_train_date)
        #print(norm_df.index[0])
        factor = max_features[target_col]

        # end_train_date needs to be lower than end_date, otherwise we will get an error in the value inside loc 
        if datetime.strptime(end_train_date, '%Y-%m-%d') < datetime.strptime(end_date, '%Y-%m-%d'):
            X_train, Y_train, X_test, Y_test = split_data(
                    norm_df.loc[norm_df.index <= end_train_date],
                    look_back= look_back,
                    ratio=1,
                    predict_n = predict_n, 
                    Y_column=target_col,
            )

            # X_pred and Y_pred will already concat the train and test array to easy the work of make 
            # the predicions of both 
            X_pred, Y_pred, X_test, Y_test = split_data(
                    norm_df,
                    look_back= look_back,
                    ratio=1,
                    predict_n = predict_n, 
                    Y_column=target_col,
            ) 

    return norm_df, factor,  X_train, Y_train, X_pred, Y_pred


def get_ml_data(city, ini_date, end_train_date, end_date, ratio, predict_n, look_back, filename = None):

    data = pd.read_csv(filename, index_col = 'Unnamed: 0' )
    data.index = pd.to_datetime(data.index)  

    data = data.dropna()

    target = f'casos_{city}'

    data_lag = build_lagged_features(data, look_back)
    data_lag.dropna()
    
    if ini_date != None:
        data_lag = data_lag.loc[ini_date: ]

    if end_date != None:
        data_lag = data_lag.loc[:end_date]

    X_data = data_lag.copy()

    # Let's remove the features not lagged from X_data to be sure that we are using just
    # past information in the forecast
    drop_columns = []
    for i in X_data.columns:
        if 'lag' not in i: 
            drop_columns.append(i)

    X_data = X_data.drop(drop_columns, axis=1)

    targets = {}
    for d in range(1, predict_n + 1):
        targets[d] = data_lag[target].shift(-(d))[:-(d)]

    if end_train_date == None: 
        X_train, X_test, y_train, y_test = train_test_split(X_data, data_lag[target],
                                                        train_size = ratio, test_size = 1 -ratio, shuffle=False)
    
    else: 
        X_train = X_data.loc[:end_train_date]

    return X_data, X_train, targets, data_lag[target] 










