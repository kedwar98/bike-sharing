
@author: rachelschaefer
"""

import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
import missingno as msno
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import warnings
from itertools import groupby  
from itertools import zip_longest 
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline


import re

#%%
#Import raw data
DCdf = pd.read_excel('/Users/rachelschaefer/Documents/CLIMATEDATA.xlsx', sheet_name = 0)       #sheet indices start at 0
NYCdf = pd.read_excel('/Users/rachelschaefer/Documents/CLIMATEDATA.xlsx', sheet_name = 1)
LAdf = pd.read_excel('/Users/rachelschaefer/Documents/CLIMATEDATA.xlsx', sheet_name = 2)
Bostondf = pd.read_excel('/Users/rachelschaefer/Documents/CLIMATEDATA.xlsx', sheet_name = 3)
Phildf = pd.read_excel('/Users/rachelschaefer/Documents/CLIMATEDATA.xlsx', sheet_name = 4)
Chicdf = pd.read_excel('/Users/rachelschaefer/Documents/CLIMATEDATA.xlsx', sheet_name = 5)


#%%

def weather_data_proc(data):
#Remove all but last duplicate of hours (when multiple reports are made in one hour - taking the last maximizes the time between reports
    datanew=data.copy()       #can be important to make copies so you aren't editing the original ('=' usually means a reference to the same thing, not a new copy)
    datanew['DATE'] = pd.to_datetime(datanew['DATE'])
    datanew['DATE'] = datanew['DATE'].dt.floor('h')    #truncate to hours
    datanew=datanew[~datanew['DATE'].duplicated(keep='last')]
    return datanew

NYCdfex=weather_data_proc(NYCdf)
Bostondfex=weather_data_proc(Bostondf)
DCdfex=weather_data_proc(DCdf)
LAdfex=weather_data_proc(LAdf)
Phildfex=weather_data_proc(Phildf)
Chicdfex=weather_data_proc(Chicdf)

examplediffs=set(Bostondfex['DATE']) ^ set(NYCdfex['DATE'])
#Note not every hour has a weather report - the above line can give insight into differences between datasets

#%%

def resampleweather(data):
#Fill in missing hours as new rows with 0 for the DATE column (change fillna argument for other replacement)
    datadf=data.copy()
    datadf.index=range(datadf.shape[0])
    datadf=datadf.set_index(pd.to_datetime(datadf['DATE']))
    datadf=datadf.resample('H').first().fillna(0)    #resample hours
    return datadf



#%%

def colpick(data):
#Pick out columns to use for models
#Interpolate across previously defined missing hours
    datacols=data[['STATION','DATE','HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation','HourlyRelativeHumidity','HourlyVisibility','HourlyWetBulbTemperature','HourlyWindGustSpeed','HourlyWindSpeed']]
    for col in datacols:
        if col != 'DATE':
            datacols[col] = pd.to_numeric(datacols[col], errors='ignore')
            datacols[col]=datacols[col].interpolate(method='linear')
    
    return datacols



#%%

def trim_year(data):
#Trim data to 2019
    
    datadf=data.copy()
    datadf=datadf.reset_index(drop=True) 
    datadf=datadf.drop(index=datadf.index[8760:8785],axis=0)
    return datadf




#%%

def numstrings(data):
#Replace "T" (for trace amount) with 0
#Remove "s" (for suspicious), leaving behind the number
#Replace "nan" with 0

    datadf=data.copy()
    #datadf['HourlyPrecipitation'].replace('T', 0,inplace=True)
    for col in datadf:
        datadf[col].replace('T',0,inplace=True)
        for ind in datadf.index:
    #print(type(x))
    #print(type(x))
            if isinstance(datadf[col].iloc[ind],str):

                if "s" in datadf[col].iloc[ind]:
           # print(x)
                    #print(datadf[col].iloc[ind])
                    #print(ind)
            #print(philnew.HourlyPrecipitation.iloc[ind])
                    datadf[col].iloc[ind]=datadf[col].iloc[ind][0:-1]
                    
                if "V" in datadf[col].iloc[ind]:
           # print(x)
                    print(datadf[col].iloc[ind])
                    print(ind)
            #print(philnew.HourlyPrecipitation.iloc[ind])
                    datadf[col].iloc[ind]=datadf[col].iloc[ind][0:-1]  
                    
    datadf=datadf.fillna(0)
    return datadf

bostupd=numstrings(bostupd)
laupd=numstrings(laupd)
philupd=numstrings(philupd)
dcupd=numstrings(dcupd)
nycupd=numstrings(nycupd)
chicupd=numstrings(chicupd)


#%%

bostupd.to_csv('Documents\projectsheets\Bostonweather_upd.csv', index = False)
dcupd.to_csv('Documents\projectsheets\DCweather_upd.csv', index = False)
laupd.to_csv('Documents\projectsheets\LAweather_upd.csv', index = False)
chicupd.to_csv('Documents\projectsheets\Chicagoweather_upd.csv', index = False)
philupd.to_csv('Documents\projectsheets\Philadelphiaweather_upd.csv', index = False)
nycupd.to_csv('Documents\projectsheets\\NYCweather_upd.csv', index = False)




#%%
#(Updated spreadsheets)
# Bostondata=pd.read_csv('/Users/rachelschaefer/Documents/projectsheets/Bostondata.csv')
# NYCdata=pd.read_csv('/Users/rachelschaefer/Documents/projectsheets/NYCdata.csv')
# LAdata=pd.read_csv('/Users/rachelschaefer/Documents/projectsheets/LAdata.csv')
# DCdata=pd.read_csv('/Users/rachelschaefer/Documents/projectsheets/DCdata.csv')
# Phildata=pd.read_csv('/Users/rachelschaefer/Documents/projectsheets/Philadelphiadata.csv')
# Chicdata=pd.read_csv('/Users/rachelschaefer/Documents/projectsheets/Chicagodata.csv')


