import sys
import glob
import numpy as np
from scipy.stats import percentileofscore
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from datetime import datetime, timedelta, date
import epiweeks
import geopandas as gpd
from flexitext import flexitext
from lstm import evaluate
from preprocessing import normalize_data

dfs = pd.read_csv('../data/macro_saude.csv')


def get_geocodes_and_state(macro):
    '''
    This function is used to get the geocodes and state that refer to a specific health macro region code
    
    :param macro:int. A four digit number
        
    '''

    dfs = pd.read_csv('../data/macro_saude.csv')

    geocodes = dfs.loc[dfs.code_macro == macro].geocode.unique()
    state = dfs.loc[dfs.code_macro == macro].state.values[0]

    return geocodes, state


def split_data_for(df, look_back=12, ratio=0.8, predict_n=5, Y_column=0, batch_size  = 4):
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

    s = get_next_n_weeks(ini_date=str(df.index[-1])[:10], next_days=predict_n)

    df = pd.concat([df, pd.DataFrame(index=s)])

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
    # print(train_size)

    # We are predicting only column 0    
    X_for = data[-1:, :look_back, ]

    # print(X_for.shape)

    # print(X_for[:,:, Y_column])

    return X_for


def get_nn_data_for(city, ini_date=None, end_date=None, look_back=4, predict_n=4, filename=None, batch_size = 4, 
                     end_train_date = '2024-04-21'):
    """
    :param city: int. The ibge code of the city, it's a seven number code 
    :param ini_date: string or None. Initial date to use when creating the train/test arrays 
    :param end_date: string or None. Last date to use when creating the train/test arrays
    :param end_train_date: string or None. Last day used to create the train data 
    :param ratio: float. If end_train_date is None, we use the ratio to spli the data into train and test 
    :param look_back: int. Number of last days used to make the forecast
    :param predict_n: int. Number of days forecast

    """
    df = pd.read_csv(filename, index_col='Unnamed: 0')
    df.index = pd.to_datetime(df.index)

    try:
        target_col = list(df.columns).index(f"casos_{city}")
    except:
        target_col = list(df.columns).index(f"casos_est_{city}")
        
    #print(target_col) 

    df = df.dropna()

    if ini_date != None:
        df = df.loc[ini_date:]

    if end_date != None:
        df = df.loc[:end_date]

    norm_df, max_features = normalize_data(df, end_train_date = end_train_date)
    
    factor = max_features[target_col]

    X_for = split_data_for(
        norm_df,
        look_back=look_back,
        ratio=1,
        predict_n=predict_n,
        Y_column=target_col,
        batch_size=batch_size
    )

    return X_for, factor


def get_next_n_weeks(ini_date: str, next_days: int) -> list:
    """
    Return a list of dates with the {next_days} days after ini_date.
    This function was designed to generate the dates of the forecast
    models.
    Parameters
    ----------
    ini_date : str
        Initial date.
    next_days : int
        Number of days to be included in the list after the date in
        ini_date.
    Returns
    -------
    list
        A list with the dates computed.
    """

    next_dates = []

    a = datetime.strptime(ini_date, "%Y-%m-%d")

    for i in np.arange(1, next_days + 1):
        d_i = datetime.strftime(a + timedelta(days=int(i * 7)), "%Y-%m-%d")

        next_dates.append(datetime.strptime(d_i, "%Y-%m-%d").date())

    return next_dates


def apply_forecast(city, ini_date, end_date, look_back, predict_n, filename, model_name, batch_size = 1,  end_train_date = '2023-10-01'):
    # (city, ini_date = None, end_date = None, look_back = 4, predict_n = 4, filename = filename
    X_for, factor = get_nn_data_for(city,
                                    ini_date=ini_date, end_date=end_date,
                                    look_back=look_back,
                                    predict_n=predict_n,
                                    filename=filename,  end_train_date = end_train_date
                                    )
    # print(X_for.shape)
    model = keras.models.load_model(f'./saved_models/{model_name}.keras', safe_mode=False, compile=False)
    thresholds = pd.read_csv('../data/typical_inc_curves_macroregiao.csv')

    pred = evaluate(model, X_for, batch_size=batch_size)

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2)) * factor
    df_pred2_5 = pd.DataFrame(np.percentile(pred, 2.5, axis=2)) * factor
    df_pred97_5 = pd.DataFrame(np.percentile(pred, 97.5, axis=2)) * factor
    df_pred25 = pd.DataFrame(np.percentile(pred, 25, axis=2)) * factor
    df_pred75 = pd.DataFrame(np.percentile(pred, 75, axis=2)) * factor

    df = create_df_for(end_date, predict_n, df_pred, df_pred2_5, df_pred25, df_pred75, df_pred97_5)

    if len(str(city)) == 4:
        df['macroregion'] = city

    if len(str(city)) == 7:
        df['city'] = city

    prob_high = []
    prob_low = []
    HTs = []
    LTs = []
    HTinc = []
    LTinc = []

    for w, dt in enumerate(df.date):
        values = (pred[:, w, :] * factor).reshape(-1)
        SE = epiweeks.Week.fromdate(dt).week
        ht = thresholds[(thresholds.SE == SE) & (thresholds.macroregional_id == int(city))].HighCases.values[0]
        lt = thresholds[(thresholds.SE == SE) & (thresholds.macroregional_id == int(city))].LowCases.values[0]
        htinc = thresholds[(thresholds.SE == SE) & (thresholds.macroregional_id == int(city))].High.values[0]
        ltinc = thresholds[(thresholds.SE == SE) & (thresholds.macroregional_id == int(city))].Low.values[0]
        prob_high.append(100 - percentileofscore(values, ht))
        prob_low.append(percentileofscore(values, lt))
        HTs.append(ht)
        LTs.append(lt)
        HTinc.append(htinc)
        LTinc.append(ltinc)
    df['prob_high'] = prob_high
    df['prob_low'] = prob_low
    df['HT'] = HTs
    df['LT'] = LTs
    df['HTinc'] = HTinc
    df['LTinc'] = LTinc

    df.to_csv(f'./forecast_tables/forecast_{city}.csv.gz')
    return df


def plot_for(filename, region, df_for, ini_date, plot = False):

    fig, ax = plt.subplots(figsize=(8, 5))

    df_data = pd.read_csv(filename, index_col='Unnamed: 0')

    df_data.index = pd.to_datetime(df_data.index)

    ax.plot(df_data[ini_date:][f'casos_est_{region}'], label='Data', color='black')

    ax.plot(df_for.date, df_for.forecast, color='tab:red', label='Forecast')

    ax.fill_between(df_for.date, df_for.lower_2_5,
                    df_for.upper_97_5, color='tab:red', alpha=0.3)

    ax.fill_between(df_for.date, df_for.lower_25,
                    df_for.upper_75, color='tab:red', alpha=0.3)

    ax.legend()

    ax.grid()

    ax.set_title(f'{region}')

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    ax.set_ylabel('Forecast de casos notificados')

    ax.set_xlabel('Data')

    plt.savefig(f"./plots/predictions/forecast_{region}.png", dpi=300, bbox_inches='tight')

    if plot:
        plt.show()


def create_df_for(ini_date_for, predict_n, df_pred, df_pred2_5, df_pred25, df_pred75, df_pred975):
    df = pd.DataFrame()

    for_dates = get_next_n_weeks(f'{ini_date_for}', predict_n)

    df['date'] = for_dates
    df['lower_2_5'] = df_pred2_5.iloc[-1].values
    df['lower_25'] = df_pred25.iloc[-1].values
    df['forecast'] = df_pred.iloc[-1].values
    df['upper_75'] = df_pred75.iloc[-1].values
    df['upper_97_5'] = df_pred975.iloc[-1].values

    return df


def plot_for_macro(macro, df_for, df_muni, ini_date='2023-01-01', filename=None, plot=False):
    geocodes, state = get_geocodes_and_state(macro)

    df_data = pd.read_csv(filename, index_col='Unnamed: 0')

    df_data.index = pd.to_datetime(df_data.index)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_data[ini_date:][f'casos_est_{macro}'], label='Data', color='black')

    ax.plot([df_data.index[-1], df_for.date[0]], [df_data[f'casos_est_{macro}'].values[-1], df_for.forecast.values[0]], ls = '--', color = 'black')

    ax.plot(df_for.date, df_for.forecast, color='tab:red', label='Forecast')

    ax.fill_between(df_for.date, df_for.lower_2_5,
                    df_for.upper_97_5, color='tab:red', alpha=0.3)

    ax.fill_between(df_for.date, df_for.lower_25,
                    df_for.upper_75, color='tab:red', alpha=0.3)

    ax.legend(loc='upper right')

    ax.grid()

    name_macro = dfs.loc[dfs.code_macro == macro].name_macro.values[0]

    ax.set_title(f'{name_macro} - {state}')

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    ax.set_ylabel('Forecast de casos notificados')

    ax.set_xlabel('Data')

    l, b, h, w = .07, .7, .15, .3
    ax2 = fig.add_axes([l, b, w, h])

    df_muni.loc[df_muni.abbrev_state == state].plot(ax=ax2, color='lightgray')

    df_muni.loc[df_muni.code_muni.isin(geocodes)].plot(ax=ax2, color='red')

    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.savefig(f"./plots/predictions/forecast_{name_macro.replace('/', '-')}_{state}.png", dpi=300, bbox_inches='tight')

    if plot:
        plt.show()


def apply_forecast_macro(macro, ini_date, end_date, look_back, predict_n, filename, model_name, df_muni, plot=False, batch_size = 1, end_train_date = '2024-04-21'):
    df_for = apply_forecast(macro, ini_date, end_date, look_back=look_back, predict_n=predict_n, filename=filename,
                            model_name=model_name, batch_size=batch_size, end_train_date=end_train_date)

    plot_for_macro(int(macro), df_for, df_muni, ini_date='2023-01-01', filename=filename, plot=plot)

    return df_for


def plot_prob_map(week_idx):
    # loading all macro forcasts on a single dataframe
    for i, m in enumerate(glob.glob('./forecast_tables/forecast_[0-9][0-9][0-9][0-9].csv.gz')):
        if i == 0:
            df = pd.read_csv(m)
            dates = df.date.unique()
        else:
            df = pd.concat([df, pd.read_csv(m)])

    df.prob_low = -df.prob_low
    df['prob_color'] = df.apply(lambda x: x.prob_low if abs(x.prob_low) > abs(x.prob_high) else x.prob_high, axis=1)
    df['prob_color'] = df.prob_color.apply(lambda x: 0 if abs(x) < 50 else x)

    df_macros = pd.read_csv('../data/macro_saude.csv')
    df_muni = gpd.read_file('../data/muni_br.gpkg')
    df_muni = df_muni.merge(df_macros[['geocode', 'code_macro']], left_on='code_muni', right_on='geocode', how='left')
    # df_muni['macro'] = df_muni.apply(lambda x: df_macros.loc[df_macros.geocode == x.code_muni].code_macro.values[0],
    #                                  axis = 1)
    df_muni = df_muni.merge(df[df.date == dates[week_idx]], left_on='code_macro', right_on='macroregion', how='left')
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))
    df_muni.plot(ax=ax1, column='HTinc',
                 cmap='viridis',
                 legend=True, figsize=(10, 10), legend_kwds={'label': "Incidência /100.000 hab.", 
                             "shrink":.55})
    df_muni.plot(ax=ax2, column='prob_color',
                 cmap='coolwarm', vmin=-100, vmax=100,
                 legend=True, figsize=(10, 10),
                 legend_kwds={'label': "Probabilidade (%)", 
                             "shrink":.55})
    ax2.set_axis_off()
    ax1.set_axis_off()
    ax2.set_title('Previsão probabilística na semana de ' + str(dates[week_idx])[:10])
    ax1.set_title('Limiar superior de Incidência na semana de ' + str(dates[week_idx])[:10])

    df_states = gpd.read_file('../data/states.gpkg')

    #df_states = df_states.dropna()
    df_states.drop(['id'], axis =1, inplace = True)
    df_states['SIGLA'] = df_states['codarea'].astype(int).replace(code_to_state)

    df_states.boundary.plot(ax =ax1,color = 'black')
    df_states.boundary.plot(ax =ax2,color = 'black')

    text = "<size:12> <color:royalblue, weight:bold>Azul</>: abaixo do limiar inferior \n <color:crimson, weight:bold>Vermelho</>: acima do limiar superior \n Cinza: compatível com a mediana \n histórica </>"
    flexitext(0.16, 0.255, text, ha="center")

    #ax2.text(0.1, -0.02, 'Regiões em cinza, representam previsão compatível com a mediana histórica\n Azul: abaixo do limiar inferior\n Vermelho: acima do limiar superior',
     #        transform=ax2.transAxes, fontsize='x-small')
    
    plt.subplots_adjust(wspace = 0.0)
    plt.savefig(f'./plots/prob_map_{dates[week_idx]}.png', dpi=300, bbox_inches='tight')

code_to_state = {33: 'RJ', 32: 'ES', 41: 'PR', 23: 'CE', 21: 'MA',
 31: 'MG', 42: 'SC', 26: 'PE', 25: 'PB', 24: 'RN', 22: 'PI', 27: 'AL',
 28: 'SE', 35: 'SP', 43: 'RS', 15: 'PA', 16: 'AP', 14: 'RR',  11: 'RO',
 13: 'AM', 12: 'AC', 51: 'MT', 50: 'MS', 52: 'GO', 17: 'TO', 53: 'DF',
 29: 'BA'}

def plot_prob_map_states(week_idx):
    # loading all macro forcasts on a single dataframe
    for i, m in enumerate(code_to_state.values()):
        if i == 0:
            df = pd.read_csv(f'./forecast_tables/forecast_{m}.csv.gz', index_col = 'Unnamed: 0')
            dates = df.date.unique()
        else:
            df = pd.concat([df, pd.read_csv(f'./forecast_tables/forecast_{m}.csv.gz',  index_col = 'Unnamed: 0')])

    df.prob_low = -df.prob_low
    df['prob_color'] = df.apply(lambda x: x.prob_low if abs(x.prob_low) > abs(x.prob_high) else x.prob_high, axis=1)
    df['prob_color'] = df.prob_color.apply(lambda x: 0 if abs(x) < 50 else x)
    
    df_states = gpd.read_file('../states.gpkg')

    #df_states = df_states.dropna()
    df_states.drop(['id'], axis =1, inplace = True)
    df_states['SIGLA'] = df_states['codarea'].astype(int).replace(code_to_state)
    
    df_states = df_states.merge(df[df.date == dates[week_idx]], left_on='SIGLA', right_on='state', how='left')


    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))
    df_states.plot(ax=ax1, column='HTinc',
                 cmap='viridis',
                 legend=True, figsize=(10, 10), legend_kwds={'label': "Incidência /100.000 hab.",
                                                             "shrink":.55})
    df_states.plot(ax=ax2, column='prob_color',
                 cmap='coolwarm', vmin=-100, vmax=100,
                 legend=True, figsize=(10, 10),
                 legend_kwds={'label': "Probabilidade (%)",
                              #'ticks': [100, 50, 0, 50, 100], 
                             "shrink":.55})
    

    ax2.set_axis_off()
    ax1.set_axis_off()
    ax2.set_title('Previsão probabilística na semana de ' + str(dates[week_idx])[:10], fontsize = 16)
    ax1.set_title('Limiar superior de Incidência na semana de ' + str(dates[week_idx])[:10], fontsize = 16)
    
    
    text = "<size:12> <color:royalblue, weight:bold>Azul</>: abaixo do limiar inferior \n <color:crimson, weight:bold>Vermelho</>: acima do limiar superior \n Cinza: compatível com a mediana \n histórica </>"
    flexitext(0.16, 0.255, text, ha="center")

    df_states.boundary.plot(ax =ax1,color = 'black')
    df_states.boundary.plot(ax =ax2,color = 'black')

    #ax2.text(0.1, -0.02, 'Regiões em cinza, representam previsão compatível com a mediana histórica\n Azul: abaixo do limiar inferior\n Vermelho: acima do limiar superior',
             #transform=ax2.transAxes, fontsize='x-small')
        
    
    plt.subplots_adjust(wspace = 0.0)
    plt.savefig(f'./plots/prob_map_states_{dates[week_idx]}.png', dpi=300, bbox_inches='tight')


def apply_forecast_state(state, ini_date, end_date, look_back, predict_n, filename, model_name, gen_fig = True, save = True, batch_size = 1,  end_train_date = '2023-12-31'):
    # (city, ini_date = None, end_date = None, look_back = 4, predict_n = 4, filename = filename
    
    thresholds = pd.read_csv('../data/typical_inc_curves_uf.csv')

    X_for, factor = get_nn_data_for(state,
                                    ini_date=ini_date, end_date=end_date,
                                    look_back=look_back,
                                    predict_n=predict_n,
                                    filename=filename, end_train_date = end_train_date
                                    )
    # print(X_for.shape)
    model = keras.models.load_model(f'../saved_models/{model_name}.keras',safe_mode=False, compile=False)
    
    pred = evaluate(model, X_for, batch_size=batch_size)

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2)) * factor
    df_pred2_5 = pd.DataFrame(np.percentile(pred, 2.5, axis=2)) * factor
    df_pred97_5 = pd.DataFrame(np.percentile(pred, 97.5, axis=2)) * factor
    df_pred25 = pd.DataFrame(np.percentile(pred, 25, axis=2)) * factor
    df_pred75 = pd.DataFrame(np.percentile(pred, 75, axis=2)) * factor

    df = create_df_for(end_date, predict_n, df_pred, df_pred2_5, df_pred25, df_pred75, df_pred97_5)

    df['state'] = state
    
    prob_high = []
    prob_low = []
    HTs = []
    LTs = []
    HTinc = []
    LTinc = []

    for w, dt in enumerate(df.date):
        values = (pred[:, w, :] * factor).reshape(-1)
        SE = epiweeks.Week.fromdate(dt).week
        ht = thresholds[(thresholds.SE == SE) & (thresholds.UF_id == state)].HighCases.values[0]
        lt = thresholds[(thresholds.SE == SE) & (thresholds.UF_id == state)].LowCases.values[0]
        htinc = thresholds[(thresholds.SE == SE) & (thresholds.UF_id ==state)].High.values[0]
        ltinc = thresholds[(thresholds.SE == SE) & (thresholds.UF_id == state)].Low.values[0]
        prob_high.append(100 - percentileofscore(values, ht))
        prob_low.append(percentileofscore(values, lt))
        HTs.append(ht)
        LTs.append(lt)
        HTinc.append(htinc)
        LTinc.append(ltinc)
    df['prob_high'] = prob_high
    df['prob_low'] = prob_low
    df['HT'] = HTs
    df['LT'] = LTs
    df['HTinc'] = HTinc
    df['LTinc'] = LTinc

    if save: 
        df.to_csv(f'./forecast_tables/forecast_{state}.csv.gz')

    if gen_fig: 

        plot_for( filename = filename, region = state, df_for= df, ini_date='2023-01-01', plot = False)

    return df




def apply_forecast_(city, ini_date, end_date, look_back, predict_n, filename, model_name, batch_size = 1, end_train_date = '2023-12-31'):
    # (city, ini_date = None, end_date = None, look_back = 4, predict_n = 4, filename = filename
    X_for, factor = get_nn_data_for(city,
                                    ini_date=ini_date, end_date=end_date,
                                    look_back=look_back,
                                    predict_n=predict_n,
                                    filename=filename, end_train_date = end_train_date
                                    )
    # print(X_for.shape)\

    model = keras.models.load_model(f'../saved_models/{model_name}.keras', safe_mode=False, compile=False)

    pred = evaluate(model, X_for, batch_size=batch_size)

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2)) * factor
    df_pred2_5 = pd.DataFrame(np.percentile(pred, 2.5, axis=2)) * factor
    df_pred97_5 = pd.DataFrame(np.percentile(pred, 97.5, axis=2)) * factor
    df_pred25 = pd.DataFrame(np.percentile(pred, 25, axis=2)) * factor
    df_pred75 = pd.DataFrame(np.percentile(pred, 75, axis=2)) * factor

    df = create_df_for(end_date, predict_n, df_pred, df_pred2_5, df_pred25, df_pred75, df_pred97_5)

    return df
