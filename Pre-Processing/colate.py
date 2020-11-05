"""
@author Kelli Waterman
This file is used to combine new data together, and can be used to do city data or overall data
"""

import pandas as pd
import numpy as np

def put_together(city):
    weather = pd.read_csv('cleaned_data/'+city+'weather_uptd.csv',sep=",",usecols=('HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation','HourlyRelativeHumidity','HourlyVisibility','HourlyWetBulbTemperature','HourlyWindGustSpeed','HourlyWindSpeed'))
    dates = pd.read_csv('cleaned_data/'+city+'_rph.csv',sep = ',',usecols=('month','day','hour'))
    census = pd.read_csv('cleaned_data/'+city+'_census.csv', sep = ',')

    rph = pd.read_csv('cleaned_data/'+city+'_rph.csv',sep=",", usecols=['ridesperhour'])
    new = pd.concat((dates,weather,census,rph),axis=1)

    pd.DataFrame.to_csv(new, city+"data.csv", sep = ",")

    return(new)

put_together('Boston')
put_together('DC')
put_together('Philadelphia')
put_together('NYC')
put_together('LA')
put_together('Chicago')

def total():
    a = np.vstack((put_together('Boston'),put_together('DC'),put_together('Philadelphia'),put_together('LA'),put_together('NYC'), put_together('Chicago')))
    np.savetxt('totaldata.csv',a, delimiter = ',')
    return()

total()
