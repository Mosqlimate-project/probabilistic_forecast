import pandas as pd
import geopandas as gpd
from forecast import apply_forecast_macro, apply_forecast_state

# from train_models_to_for import LOOK_BACK, PREDICT_N
PREDICT_N = 4  # number of new days predicted
LOOK_BACK = 8

df_muni = gpd.read_file('../data/muni_br.gpkg')

dfs = pd.read_csv('../data/macro_saude.csv')
ini_date = None
end_date = '2024-04-28'
# for macro in dfs.loc[dfs.state=='MG'].code_macro.unique():
#for macro in dfs.code_macro.unique():
for macro in ['3103']:
    print(f'Forecasting: {macro}')

    filename = f'../data/dengue_{macro}.csv.gz'
    model_name = f'trained_{macro}_dengue_macro_4'

    df_for = apply_forecast_macro(macro, ini_date, end_date, look_back=LOOK_BACK, predict_n=PREDICT_N,
                                  filename=filename, model_name=model_name, df_muni=df_muni)
    

'''
for state in dfs.state.unique():
    print(f'Forecasting: {state}')

    filename = f'../data/dengue_{state}.csv.gz'
    model_name = f'trained_{state}_dengue_state_4'

    df_for = apply_forecast_state(state, ini_date, end_date, look_back=LOOK_BACK, predict_n=PREDICT_N,
                                  filename=filename, model_name=model_name, gen_fig = True)

'''