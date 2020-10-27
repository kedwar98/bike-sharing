import pandas as pd

def put_together(city):
    weather = pd.read_csv('cleaned_data/'+city+'weather_uptd.csv',sep=",",usecols=('HourlyDewPointTemperature','HourlyDryBulbTemperature','HourlyPrecipitation','HourlyRelativeHumidity','HourlyVisibility','HourlyWetBulbTemperature','HourlyWindGustSpeed','HourlyWindSpeed'))
    dates = pd.read_csv('cleaned_data/'+city+'_rph.csv',sep = ',',usecols=('month','day','hour'))
    rph = pd.read_csv('cleaned_data/'+city+'_rph.csv',sep=",", usecols=['ridesperhour'])
    new = pd.concat((dates,weather,rph),axis=1)
    pd.DataFrame.to_csv(new, city+"data.csv", sep = ",")

    return()

put_together('Boston')
put_together('DC')
put_together('Philadelphia')
put_together('NYC')
put_together('LA')
put_together('Chicago')
