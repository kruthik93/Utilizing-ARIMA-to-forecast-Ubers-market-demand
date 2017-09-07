"""#############################################################################
Author: Kruthik Heggade
Project: Market demand prediction using ARIMA
Description: We will be using the ARIMA package available in the statsmodels 
package to predict market demand for uber in NYC. The dataset used is found below
(link:https://www.kaggle.com/fivethirtyeight/uber-pickups-in-new-york-city)
Last updated: 7th-Sep-2017
#############################################################################"""

#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

#Defining functions

#Preparing the uber 2014 main dataset
def prepare_2014_df():
    uber_2014_apr=pd.read_csv('uber-raw-data-apr14.csv',header=0)
    uber_2014_may=pd.read_csv('uber-raw-data-may14.csv',header=0)
    uber_2014_jun=pd.read_csv('uber-raw-data-jun14.csv',header=0)
    uber_2014_jul=pd.read_csv('uber-raw-data-jul14.csv',header=0)
    uber_2014_aug=pd.read_csv('uber-raw-data-aug14.csv',header=0)
    uber_2014_sep=pd.read_csv('uber-raw-data-sep14.csv',header=0)
    
    df = uber_2014_apr.append([uber_2014_may,uber_2014_jun,uber_2014_jul,uber_2014_aug,uber_2014_sep], ignore_index=True)
    return df

# Feature Engineering
def create_day_series(df):
    
    # Grouping by Date/Time to calculate number of trips
    day_df = pd.Series(df.groupby(['Date/Time']).size())
    # setting Date/Time as index
    day_df.index = pd.DatetimeIndex(day_df.index)
    # Resampling to daily trips
    day_df = day_df.resample('1D',how=np.sum)
    
    return day_df
    
#ARIMA

#Checking trend and autocorrelation
def initial_plots(time_series):

    #Original timeseries plot
    plt.figure(1)
    plt.plot(time_series)
    plt.title('Original data across time')
    plt.figure(2)
    autocorrelation_plot(time_series)
    plt.title('Autocorrelation plot')
    
#fitting ARIMA model on dataset
def ARIMA_call(time_series,p,d,q):    
    
    #fitting the model
    ARIMA_model = ARIMA(time_series.astype(float), order=(p,d,q))
    ARIMA_model_fit = ARIMA_model.fit(disp=0)
    print 'AIC = {}'.format(np.round(ARIMA_model_fit.aic,2))
    print 'BIC = {}'.format(np.round(ARIMA_model_fit.bic,2))
    fitted_values = ARIMA_model_fit.predict(1,len(time_series)-1,typ='levels') #Index starts at 1 as differencing is in effect
    #Plotting
    plt.figure(3)
    plt.title('Plot of original data and fitted values of the ARIMA({},{},{}) model'.format(p,d,q))
    plt.plot(time_series[1:],'k-',label='Original data') #Plotting from index of 1 as differencing is in effect
    plt.plot(fitted_values,'r--',label='Fitted values')
    plt.legend()
    plt.show()
    return ARIMA_model_fit,fitted_values

def predict(df,fitted_model,n_days,conf):
    #Predicting
    values = pd.DataFrame(fitted_model.forecast(steps=n_days,alpha=(1-conf))[0],columns=['Prediction'])
    values.index = pd.date_range(df.index.max()+1,periods=n_days)
    conf_int = pd.DataFrame(fitted_model.forecast(steps=n_days,alpha=(1-conf))[2],columns=['Conf_int_lower_lim','Conf_int_upper_lim'])
    conf_int.index = pd.date_range(df.index.max()+1,periods=n_days)
    ARIMA_prediction = pd.merge(values,conf_int,left_index=True,right_index=True)
    #Plotting
    plt.figure(4)
    plt.title('Plot of original data and predicted values using the ARIMA model')
    plt.xlabel('Time')
    plt.ylabel('Number of Trips')
    plt.plot(df[1:],'k-', label='Original data')
    plt.plot(values,'r--', label='Next {}days predicted values'.format(n_days))
    plt.plot(conf_int,'b--', label='{}% confidence interval'.format(round(conf*100)))
    plt.legend()
    plt.show()
    #Returning predicitons
    return ARIMA_prediction
    

#Main program

#Uber 2014 dataset

uber_2014_master = prepare_2014_df()
day_df_2014 = create_day_series(uber_2014_master)
initial_plots(day_df_2014)
fitted_model,fitted_values = ARIMA_call(day_df_2014,7,1,0)
prediction = predict(day_df_2014,fitted_model,7,0.80)