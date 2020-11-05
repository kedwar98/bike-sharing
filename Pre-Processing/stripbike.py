"""
@author Kelli Waterman
This python code is used to combine bike share data from publicly available bike share sources. 
Pre-steps necessary include renaming 'starttime' column for some cities, reformatting starttime to '%Y/%m/%d %H:%M:%S'
and sorting by date-time.
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime
import csv

#system is month or quarter depending on the city
#number is 0 for month system or quarter number (1,2,3, or 4)
def get_rph(bike_data_path, system, number):
    file_1 = pd.read_csv(bike_data_path,sep=",",usecols=['starttime'])

    current_time = dt.strptime(file_1.iloc[0,0],'%Y/%m/%d %H:%M:%S')
    if system == 'month':
        nr_days = int('{:%d}'.format(dt.strptime(file_1.iloc[-1,0],'%Y/%m/%d %H:%M:%S')))
        total = nr_days*24
    if system == 'quarter':
        if number == 1:
            total = 90*24
        if number == 2:
            total = 91*24
        if number == 3:
            total = 92*24
        if number == 4:
            total = 92*24
    final_vector = np.zeros((total,4))
    h_count = 0
    row_count = 0
    for i in range(len(file_1)):
        day_hour = dt.strptime(file_1.iloc[i,0],'%Y/%m/%d %H:%M:%S')

        if day_hour.month == current_time.month and day_hour.day == current_time.day and day_hour.hour == current_time.hour:
            h_count +=1

        else:
            final_vector[row_count] = ['{:%m}'.format(dt.strptime(file_1.iloc[i-1,0],'%Y/%m/%d %H:%M:%S')), '{:%d}'.format(dt.strptime(file_1.iloc[i-1,0],'%Y/%m/%d %H:%M:%S')), '{:%H}'.format(dt.strptime(file_1.iloc[i-1,0],'%Y/%m/%d %H:%M:%S')), h_count]

            test_time = dt.strptime(file_1.iloc[i-1,0], '%Y/%m/%d %H:%M:%S')
            test_time += datetime.timedelta(hours = 1)
            a = test_time

            while day_hour.day != test_time.day or day_hour.hour != test_time.hour:
                h_count = 0
                row_count += 1
                b = str(a)
                final_vector[row_count] = ['{:%m}'.format(dt.strptime(b,'%Y-%m-%d %H:%M:%S')),'{:%d}'.format(dt.strptime(b,'%Y-%m-%d %H:%M:%S')),'{:%H}'.format(dt.strptime(b,'%Y-%m-%d %H:%M:%S')),h_count]

                a += datetime.timedelta(hours=1)
                test_time = a
            h_count = 1
            #print(h_count)
            row_count += 1
            current_time = day_hour

    final_vector[row_count] = ['{:%m}'.format(dt.strptime(str(current_time),'%Y-%m-%d %H:%M:%S')),'{:%d}'.format(dt.strptime(str(current_time),'%Y-%m-%d %H:%M:%S')),'{:%H}'.format(dt.strptime(str(current_time),'%Y-%m-%d %H:%M:%S')),h_count]

    return(final_vector)

#Combine all the files for each city

final = np.vstack((get_rph('201901-bluebikes-tripdata.csv', 'month', 0),get_rph('201902-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201903-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201904-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201905-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201906-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201907-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201908-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201909-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201910-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201911-bluebikes-tripdata.csv','month',0)))
final = np.vstack((final,get_rph('201912-bluebikes-tripdata.csv','month',0)))


final_nyc = np.vstack((get_rph('201901-citibike-tripdata.csv','month',0),get_rph('201902-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201903-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201904-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201905-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201906-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201907-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201908-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201909-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201910-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201911-citibike-tripdata.csv','month',0)))
final_nyc = np.vstack((final_nyc,get_rph('201912-citibike-tripdata.csv','month',0)))

final_LA = np.vstack((get_rph('LA-metro-bike-share-trips-2019-q1.csv','quarter',1),get_rph('LA-metro-bike-share-trips-2019-q2.csv','quarter',2)))
final_LA = np.vstack((final_LA,get_rph('LA-metro-bike-share-trips-2019-q3.csv','quarter',3)))
final_LA = np.vstack((final_LA,get_rph('LA-metro-bike-share-trips-2019-q4.csv','quarter',4)))
#
final_DC = np.vstack((get_rph('201901-capitalbikeshare-tripdata.csv','month',0),get_rph('201902-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201903-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201904-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201905-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201906-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201907-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201908-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201909-capitalbikeshare-tripdata.csv.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201910-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201911-capitalbikeshare-tripdata.csv','month',0)))
final_DC = np.vstack((final_DC,get_rph('201912-capitalbikeshare-tripdata.csv','month',0)))

final_philly = np.vstack((get_rph('indego-trips-2019-q1.csv','quarter',1),get_rph('indego-trips-2019-q2.csv','quarter',2)))
final_philly = np.vstack((final_philly,get_rph('indego-trips-2019-q3.csv','quarter',3)))
final_philly = np.vstack((final_philly,get_rph('indego-trips-2019-q4.csv','quarter',4)))

final_chicago = np.vstack((get_rph('metro-bike-share-trips-2019-q1.csv','quarter',1),get_rph('metro-bike-share-trips-2019-q2.csv','quarter',2)))
final_chicago = np.vstack((final_chicago,get_rph('metro-bike-share-trips-2019-q3.csv','quarter',3)))
final_chicago = np.vstack((final_chicago,get_rph('metro-bike-share-trips-2019-q4.csv','quarter',4)))

#Output each city final rides per hour
np.savetxt("BOSTON.csv", final, delimiter = ",")
np.savetxt("NYC.csv", final_nyc, delimiter = ",")
np.savetxt("PHILADELPHIA.csv", final_philly, delimiter = ",")
np.savetxt("LA.csv", final_LA, delimiter = ",")
np.savetxt("DC.csv", final_DC, delimiter = ",")
np.savetxt("CHICAGO.csv", final_chicago, delimiter = ",")




