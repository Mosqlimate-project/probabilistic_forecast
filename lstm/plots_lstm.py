import numpy as np 
import pandas as pd 
import seaborn as sns
import scipy.stats as ss
from matplotlib import pyplot as P
import matplotlib.pyplot as plt 

MAIN_FOLDER = './'


def plot_train(ax, indice, Ydata, factor, df_predicted, df_predicted25, df_predicted975, split_point, label):
    
    if (type(factor) == np.float64):
        
        ax.plot(indice[len(indice)-Ydata.shape[0]:split_point +len(indice)-Ydata.shape[0]], Ydata[:split_point, -1] * factor, 'k-', alpha=0.7, label='data')
        ax.plot(indice[len(indice)-Ydata.shape[0]:split_point + len(indice)-Ydata.shape[0] ], df_predicted.iloc[:split_point,-1] * factor, 'r-', alpha=0.5, label='median')
        

        ax.fill_between(indice[len(indice)-Ydata.shape[0]:split_point+len(indice)-Ydata.shape[0]], df_predicted25[df_predicted25.columns[-1]][:split_point] * factor,
                       df_predicted975[df_predicted975.columns[-1]][:split_point] * factor,
                       color='b', alpha=0.3)
        
    else: 

        ax.plot(indice[len(indice)-Ydata.shape[0]:split_point +len(indice)-Ydata.shape[0]], factor.inverse_transform(Ydata[:split_point, -1].reshape(-1,1)).reshape(-1) , 'k-', alpha=0.7, label='data')
        ax.plot(indice[len(indice)-Ydata.shape[0]:split_point + len(indice)-Ydata.shape[0] ], factor.inverse_transform(df_predicted.iloc[:split_point,-1].values.reshape(-1,1)).reshape(-1), 'r-', alpha=0.5, label='median')
        

        ax.fill_between(indice[len(indice)-Ydata.shape[0]:split_point+len(indice)-Ydata.shape[0]], factor.inverse_transform(df_predicted25[df_predicted25.columns[-1]][:split_point].values.reshape(-1,1)).reshape(-1),
                       factor.inverse_transform(df_predicted975[df_predicted975.columns[-1]][:split_point].values.reshape(-1,1)).reshape(-1),
                       color='b', alpha=0.3)

    ax.grid()
    ax.legend()
    ax.set_title(f'Train - {label}')
    ax.set_xlabel("time")
    ax.set_ylabel("incidence")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

def plot_test(ax, indice, Ydata, factor, df_predicted, df_predicted25, df_predicted975, split_point, label):

    if (type(factor) == np.float64): 

        #print(Ydata.shape)
        #print(df_predicted.shape)
    
        ax.plot(indice[split_point + len(indice)-Ydata.shape[0]:], Ydata[split_point:, -1] * factor, 'k-', alpha=0.7, label='data')
        ax.plot(indice[split_point+ len(indice)-Ydata.shape[0]:], df_predicted.iloc[split_point:,-1] * factor, 'r-', alpha=0.5, label='median')
        

        ax.fill_between(indice[split_point + len(indice)-Ydata.shape[0]:], df_predicted25[df_predicted25.columns[-1]][split_point:] * factor,
                        df_predicted975[df_predicted975.columns[-1]][split_point:] * factor,
                        color='b', alpha=0.3)
        
    else: 

        ax.plot(indice[split_point + len(indice)-Ydata.shape[0]:], factor.inverse_transform(Ydata[split_point:, -1].reshape(-1,1)).reshape(-1), 'k-', alpha=0.7, label='data')
        ax.plot(indice[split_point+ len(indice)-Ydata.shape[0]:], factor.inverse_transform(df_predicted.iloc[split_point:,-1].values.reshape(-1,1)).reshape(-1), 'r-', alpha=0.5, label='median')
        

        ax.fill_between(indice[split_point + len(indice)-Ydata.shape[0]:], factor.inverse_transform(df_predicted25[df_predicted25.columns[-1]][split_point:].values.reshape(-1,1)).reshape(-1),
                        factor.inverse_transform(df_predicted975[df_predicted975.columns[-1]][split_point:].values.reshape(-1,1)).reshape(-1),
                        color='b', alpha=0.3)
        


    ax.grid()
    ax.legend()
    ax.set_title(f'Test - {label}')
    ax.set_xlabel("time")
    ax.set_ylabel("incidence")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)


def plot_train_test(indice, Ydata, factor, df_predicted, df_predicted25, df_predicted975, split_point, label):

    fig, ax = plt.subplots(1,2, figsize=(12,4.5))

    plot_train(ax[0], indice, Ydata, factor, df_predicted, df_predicted25, df_predicted975, split_point, label)

    plot_test(ax[1], indice, Ydata, factor, df_predicted, df_predicted25, df_predicted975, split_point, label)

    plt.savefig(f'{MAIN_FOLDER}/plots/predictions/{label}.png',bbox_inches='tight',  dpi = 300)

    plt.show()

    return 


def plot_transf_predicted_vs_data(df_predicted_t, df_predicted, Ydata, indice, label, pred_window, factor, split_point=None, uncertainty=False, label_name = 'transf', label1 = 'transf', label2 = 'chik data'):
    """
    Plot the model's predictions against data
    :param predicted: model predictions
    :param Ydata: observed data
    :param indice:
    :param label: Name of the locality of the predictions
    :param pred_window:
    :param factor: Normalizing factor for the target variable
    """

    P.clf()
        
    uncertainty = True
    ymax = max( max(df_predicted.max()) * factor, Ydata.max() * factor, max(df_predicted_t.max()) * factor)

    if split_point != None:
        P.vlines(indice[split_point], 0, ymax, "g", "dashdot", lw=2, label = 'Train/Test')
    #P.text(indice[split_point + 2], 0.6 * ymax, "Out of sample Predictions")
    # plot only the last (furthest) prediction point
    P.plot(indice[len(indice)-Ydata.shape[0]:], Ydata[:, -1] * factor, 'k-', alpha=0.7, label='data')
    P.plot(indice[len(indice)-Ydata.shape[0]:], df_predicted.iloc[:,-1] * factor, 'r-', alpha=0.5, label=label1)
    P.plot(indice[len(indice)-Ydata.shape[0]:], df_predicted_t.iloc[:,-1] * factor, 'g-', alpha=0.5, label=label2)
   
                                  
    tag = '_unc' if uncertainty else ''
    P.grid()
    #P.title("Predictions for {}".format(label))
    P.xlabel("time")
    P.ylabel("incidence")
    P.xticks(rotation=30)
    P.legend()
    P.savefig(f'{MAIN_FOLDER}/plots/predictions/{label_name}.png',bbox_inches='tight',  dpi = 300)
    P.show()

def predicted_vs_observed(predicted, real, city, state, doenca, model_name, city_name, plot=True):
    """
    Plot QQPlot for prediction values
    :param plot: generates an saves the qqplot when True (default)
    :param predicted: Predicted matrix
    :param real: Array of target_col values used in the prediction
    :param city: Geocode of the target city predicted
    :param state: State containing the city
    :param look_back: Look-back time window length used by the model
    :param all_predict_n: If True, plot the qqplot for every week predicted
    :return:
    """
    # Name = get_city_names([city])
    # data = get_alerta_table(city, state, doenca=doenca)

    obs_preds = np.hstack((predicted, real))
    q_p = [ss.percentileofscore(obs_preds, x) for x in predicted]
    q_o = [ss.percentileofscore(obs_preds, x) for x in real]
    plot_cross_qq(city, doenca, q_o, q_p, model_name, city_name)
    return np.array(q_o), np.array(q_p)


def plot_cross_qq(city, doenca, q_o, q_p,model_name, city_name):
    ax = sns.kdeplot(q_o[len(q_p) - len(q_o):], q_p, shade=True)
    ax.set_xlabel('observed')
    ax.set_ylabel('predicted')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    P.plot([0, 100], [0, 100], 'k')
    #P.title(f'Transfer prediction percentiles with {model_name.lower()} for {doenca} at {city_name}')
    P.savefig(f'{MAIN_FOLDER}/plots/lstm/cross_qqbplot_{model_name}_{doenca}_{city}.png', dpi=300)


def plot_loss(hist, title = 'Model loss'):
    """
    :params hist:
    :params title: string. Title to the plot 
    """ 
    # "Loss"
    P.plot(hist.history['loss'])
    try:
        P.plot(hist.history['val_loss'])
    except: pass 
    
    P.title(f'{title}')
    P.ylabel('loss')
    P.xlabel('epoch')
    P.legend(['train', 'validation'], loc='upper left')
    P.grid()
    P.show()


def plot_comp(df_1_train, df_2_train, df_1_val, df_2_val, metric = 'mean_squared_error'):
    """
    Function to compare the errors in train and validation dataset of models trained with
    different loss functions. 
    """
    fig, ax = P.subplots(1,2, figsize = (12,5))

    # set width of bar
    barWidth = 0.25
    # Set position of bar on X axis
    br1 = 1
    br2 = [1 + barWidth]
    
    # set height of bar
    msle_train = sum(df_1_train.loc[df_1_train.index == metric].values[0])
    c_msle_train = sum(df_2_train.loc[df_2_train.index == metric].values[0])
    
    msle_val = sum(df_1_val.loc[df_1_val.index == metric].values[0])
    c_msle_val = sum(df_2_val.loc[df_2_val.index == metric].values[0])

    # Make the plot
    ax[0].bar(br1, msle_train, color ='r', width = barWidth,
            edgecolor ='grey', label ='msle')
    ax[0].bar(br2, c_msle_train, color ='g', width = barWidth,
            edgecolor ='grey', label ='custom_msle')


    ax[0].set_ylabel('Error')
    #plt.xticks([r + barWidth for r in range(len(train))] )
    ax[0].set_title(f'{metric} - Train') 
    ax[0].legend()
    ax[0].grid()
    
    
    # Make the plot
    ax[1].bar(br1, msle_val, color ='r', width = barWidth,
            edgecolor ='grey', label ='msle')
    ax[1].bar(br2, c_msle_val, color ='g', width = barWidth,
            edgecolor ='grey', label ='custom_msle')


    ax[1].set_ylabel('Error')
    #plt.xticks([r + barWidth for r in range(len(train))] )
    ax[1].set_title(f'{metric} - Validation') 
    ax[1].legend()
    ax[1].grid()
    
    P.show()
