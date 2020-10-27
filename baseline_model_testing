
@author: rachelschaefer
"""





#%%
#Root mean square logarithmic error      (can also use sklearn.metrics.mean_squared_log_error)
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#%%
#Shuffle

#change city variable
dfb=nycupd.copy()


from sklearn.model_selection import KFold


#randomly shuffle data, as we are predicting rides per hour and not explicitly forecasting
dfb = dfb.sample(frac=1).reset_index(drop=True)
#could also use sklearn.model_selection.KFold

dfbk=KFold(n_splits=5,shuffle=True, random_state=None)

#%%

for train_index, test_index in dfbk.split(dfb):
     print("TRAIN:", train_index, "TEST:", test_index)



#%%
#make an 80:20 split out of 8760 rows
testdata=dfb[7000:-1]
datatrain=dfb[0:6999]
datatrainall=dfb[0:6999]
testdataall=dfb[7000:-1]

#%%

#Choose which features to keep or drop in data

#datatrain=philnew.copy()
dataall=dfb.copy()
dataall.drop(['ridesperhour','HourlyPrecipitation','HourlyWetBulbTemperature','HourlyWindSpeed','month'],axis=1,inplace=True)
dataall.drop(datatrain.columns[datatrain.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
rphall=dfb.copy()


#datatrain.drop(['ridesperhour','HourlyVisibility','HourlyPrecipitation','HourlyDewPointTemperature','HourlyWindSpeed','HourlyWetBulbTemperature'],axis=1,inplace=True)
datatrain.drop(['ridesperhour','HourlyPrecipitation','HourlyWetBulbTemperature','HourlyWindSpeed','month'],axis=1,inplace=True)
datatrain.drop(datatrain.columns[datatrain.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)



testdata.drop(['ridesperhour','HourlyPrecipitation','HourlyWetBulbTemperature','HourlyWindSpeed','month'],axis=1,inplace=True)
testdata.drop(testdata.columns[testdata.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#%%
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41
rfModel = RandomForestRegressor(n_estimators=1000)
yLabelsLog = np.log1p(datatrainall['ridesperhour'])
rfModel.fit(datatrain,yLabelsLog)

#training data model predictions using 80-20 split
preds = rfModel.predict(X= datatrain)
print ("RMSLE Value For Random Forest: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))

#fModel.fit(traindata,yLabelsLog)



#%%

#k-fold cross-validation using MSLE
cv = cross_validate(rfModel, dataall, rphall['ridesperhour'], scoring=metrics.make_scorer(metrics.mean_squared_log_error), cv=10)

print(cv['test_score'])



#test data model predictions using 80-20 split
testp=rfModel.predict(X=testdata)
yLabelsLog = np.log1p(testdataall['ridesperhour'])

print ("RMSLE Value For Random Forest: ",rmsle(np.exp(yLabelsLog),np.exp(testp),False))



#interesting parameter that lets you see relative contributions of features to predictions
print(rfModel.feature_importances_)

#%%

gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41
yLabelsLog = np.log1p(yLabels)
gbm.fit(dataTrain,yLabelsLog)
preds = gbm.predict(X= dataTrain)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))

testp=gbm.predict(X=testdata)
yLabelsLog = np.log1p(testdataall['ridesperhour'])

print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))

print(gbm.feature_importances_)


#%%



from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logistic regression model
lModel = LinearRegression()
yLabelsLog = np.log1p(datatrainall['ridesperhour'])

# Train the model
#yLabelsLog = np.log1p(yLabels)
lModel.fit(X = datatrain,y = yLabelsLog)

# Make predictions
preds = lModel.predict(X= datatrain)
print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))



#%%

#Neural network baseline
from keras.models import Sequential
from keras.layers import Dense








#example only
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))





#%%
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sn.distplot(yLabels,ax=ax1,bins=50)
sn.distplot(np.exp(predsTest),ax=ax2,bins=50)
