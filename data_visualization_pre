
@author: rachelschaefer
"""

#%%
#Updated spreadsheets
Bostondf=pd.read_csv('Documents\projectsheets\Bostondata.csv')
LAdf=pd.read_csv('Documents\projectsheets\LAdata.csv')
DCdf=pd.read_csv('Documents\projectsheets\DCdata.csv')
Phildf=pd.read_csv('Documents\projectsheets\Philadelphiadata.csv')
NYCdf=pd.read_csv('Documents\projectsheets\\NYCdata.csv')
Chicdf=pd.read_csv('Documents\projectsheets\Chicagodata.csv')

#%%

#correlation map
#replace "Bostondata" with other data

Bostcorr=Bostondata.corr()
mask = np.array(Bostcorr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(Bostcorr, mask=mask,vmax=.8, square=True,annot=True)



#%%Below are various plots to try. Feel free to look at relationships between the variables

#%%
plt.plot(Bostondata['hour'], Bostondata['ridesperhour'], 'o', color='black');
plt.xlabel('Hourly wind speed (mph)')
plt.ylabel('Bikeshare rides per hour')
plt.title('Boston')

#%%
plt.plot(Chicagodata['HourlyWetBulbTemperature'], Chicagodata['ridesperhour'], 'o', color='black');
plt.xlabel('Hourly wind speed (mph)')
plt.ylabel('Bikeshare rides per hour')
plt.title('Chicago')

#%%

plt.plot(LAdata['HourlyWindSpeed'], LAdata['ridesperhour'], 'o', color='black');
plt.xlabel('Hourly wind speed (mph)')
plt.ylabel('Bikeshare rides per hour')
plt.title('Los Angeles')

#%%

plt.plot(DCdata['HourlyWindSpeed'], DCdata['ridesperhour'], 'o', color='black');
plt.xlabel('Hourly wind speed (mph)')
plt.ylabel('Bikeshare rides per hour')
plt.title('DC')

#%%
#print(type(phild.ridesperhour))


plt.plot(philnew.hour,philnew.ridesperhour,'o',color='black');
plt.xlabel('Hour')
plt.ylabel('Bikeshare rides per hour')
plt.title('Philadelphia')


ax=plt.gca()
xmin, xmax = ax.get_xlim()
custom_ticks = np.linspace(xmin, xmax, 20, dtype=int)
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_ticks)

#ax.xaxis.set_major_locator(plt.MaxNLocator(20))
#%%
philnew.plot(x='HourlyPrecipitation',y='ridesperhour')

#%%
sn.scatterplot(philnew.HourlyPrecipitation,philnew.ridesperhour)

#%%
plt.plot(philnew.HourlyPrecipitation,philnew.ridesperhour)

