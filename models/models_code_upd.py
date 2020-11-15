


#%%

#Use "Run cell" to run an individual cell
#You can use "#%%" to create a cell

#Run this cell for import statements



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
#matplotlib inline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import eli5
from eli5.sklearn import PermutationImportance
import math



#%% Mean squared error

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

#mean_squared_error(y_true, y_pred, squared=False)
#Set squared=False FOR RMSE

#%%
#COMBINED
totaldata = pd.read_csv('WEATHER-CENSUS/totaldata.csv')
dfb=totaldata.copy()

#randomly shuffle data, as we are predicting rides per hour and not explicitly forecasting
dfb = dfb.sample(frac=1).reset_index(drop=True)
#could also use sklearn.model_selection.KFold

dfbk=KFold(n_splits=5,shuffle=True, random_state=None)

#make an 80:20 split out of 8760x6 rows
datatrain=dfb.copy()
testdata=dfb.copy()
testdata=dfb[42048:-1]
datatrain=dfb[0:42048]
datatrainall=dfb[0:42048]
testdataall=dfb[42048:-1]


# #%%
# #CITY
# DCdata = pd.read_csv("WEATHER/DC_norm.csv")
# dfb=DCdata.copy()    #change city name here
#
# #randomly shuffle data, as we are predicting rides per hour and not explicitly forecasting
# dfb = dfb.sample(frac=1).reset_index(drop=True)
# #could also use sklearn.model_selection.KFold
#
# dfbk=KFold(n_splits=5,shuffle=True, random_state=None)
#
# #make an 80:20 split out of 8760 rows
# datatrain=dfb.copy()
# testdata=dfb.copy()
# testdata=dfb[7008:-1]
# datatrain=dfb[0:7008]
# datatrainall=dfb[0:7008]
# testdataall=dfb[7008:-1]


#%%
#Can use this to print k_fold splits

for train_index, test_index in dfbk.split(dfb):
     print("TRAIN:", train_index, "TEST:", test_index)


#%%

#Drop ridesperhour from input data
#Drop unnamed indexing date column
#You can use the drop function to remove features for testing
#Remove census data for variables with "wo" suffix (without)


#Unsplit (Keras will do validation splits)
dataall=dfb.copy()
rphall = dataall['ridesperhour']
dataall.drop('ridesperhour',axis=1,inplace=True)
dataall.drop(datatrain.columns[datatrain.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

dataallwo=dataall.copy()
dataallwo=dataallwo.iloc[:,0:11]




#Use below variables for predefined training and testing splits

datatrainwo=datatrainall.copy()
datatrainwo.drop('ridesperhour',axis=1,inplace=True)
datatrainwo.drop(datatrain.columns[datatrain.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
datatrainwo=datatrainwo.iloc[:,0:11]

#datatrain.drop(['ridesperhour','HourlyVisibility','HourlyPrecipitation','HourlyDewPointTemperature','HourlyWindSpeed','HourlyWetBulbTemperature'],axis=1,inplace=True)
datatrain.drop('ridesperhour',axis=1,inplace=True)
datatrain.drop(datatrain.columns[datatrain.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
datatrainall.drop(datatrainall.columns[datatrainall.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


testdatawo=testdataall.copy()
testdatawo.drop('ridesperhour',axis=1,inplace=True)
testdatawo.drop(testdatawo.columns[testdatawo.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
testdatawo=testdatawo.iloc[:,0:11]

testdata.drop('ridesperhour',axis=1,inplace=True)
testdata.drop(testdata.columns[testdata.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


#%%

#Can use this for normalization / data transformation

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(datatrain)
datatraintransform=scaler.transform(datatrain)

datatesttransform=scaler.transform(testdata)





#%%

# #Random Forest Model from sklearn

#Here you can change number of trees, add other arguments, and change verbosity (2 tells you progress of tree building)
rfModel = RandomForestRegressor(n_estimators=1000,verbose=1)



#yLabelsLogall = np.log1p(datatrainall['ridesperhour'])
yLabels = datatrainall['ridesperhour']


rfModel.fit(datatrainwo,yLabels)

#training data model predictions using 80-20 split
preds = rfModel.predict(X=datatrainwo)
for i in range(len(preds)): #change
     if preds[i] < 0:
          preds[i] = 0
#Change datatrainwo to datatrainwo to include census data

print("RF Training RMSLE: ",math.sqrt(mean_squared_log_error(yLabels, preds)))
print("RF Training RMSE:",mean_squared_error(yLabels,preds,squared=False))
print("RF Training Normalized RMSE: ",mean_squared_error(yLabels, preds,squared = False)/(max(yLabels)-min(yLabels)))


#test data model predictions using 80-20 split
testp=rfModel.predict(X=testdatawo)
#Change testdatawo to testdatawi to include census data
for i in range(len(testp)):
     if testp[i] < 0:
          testp[i] = 0

yLabelstest = testdataall['ridesperhour']

print("RF Testing RMSLE: ",math.sqrt(mean_squared_log_error(yLabelstest, testp)))
print("RF Testing RMSE:",mean_squared_error(yLabelstest, testp,squared=False))
print("RF Testing Normalized RMSE: ",mean_squared_error(yLabelstest, testp, squared=False)/(max(yLabels)-min(yLabels)))


#interesting parameter that lets you see relative contributions of features to predictions
print(rfModel.feature_importances_)

#%%
#Plot RF feature importance
rffi=rfModel.feature_importances_
indices = np.argsort(rffi)[::-1]

plt.barh(range(len(indices)), rffi[indices], color='b', align='center')
plt.yticks(range(len(indices)), [dataall.columns[i] for i in indices])

plt.show()


#%%

#Gradient Boosting Model from sklearn

#Here you can change the number of estimators, the L1 regularization coefficient alpha, and other arguments, as well as verbosity
gbModel = GradientBoostingRegressor(n_estimators=1000,alpha=0.01,verbose=0)
yLabels = datatrainall['ridesperhour']

yLabelsLog = np.log1p(yLabels)
gbModel.fit(datatrainwo,yLabels)
#Change datatrainwo to datatrain to include census data

preds = gbModel.predict(X= datatrainwo)
for i in range(len(preds)):
     if preds[i] < 0:
          preds[i] = 0


print ("GB Training RMSE: ",mean_squared_error(yLabels, preds, squared=False))
print ("GB Training RMSLE: ",math.sqrt(mean_squared_log_error(yLabels, preds)))
print("GB Training Normalized RMSE: ",mean_squared_error(yLabels, preds, squared=False)/(max(yLabels)-min(yLabels)))

testp=gbModel.predict(X=testdatawo)
#Change testdatawo to testdatawi to include census data
for i in range(len(testp)):
     if testp[i] < 0:
          testp[i] = 0
yLabelstest = testdataall['ridesperhour']

print ("GB Testing RMSE: ",mean_squared_error(yLabelstest, testp, squared=False))
print("GB Testing RMSLE:", math.sqrt(mean_squared_log_error(yLabelstest, testp)))

print ("GB Testing Normalized RMSE: ",mean_squared_error(yLabelstest, testp, squared=False)/(max(yLabelstest)-min(yLabelstest)))

print(gbModel.feature_importances_)




#%%
#Plot GB feature impoptance
gbfi=gbModel.feature_importances_
indices = np.argsort(gbfi)[::-1]

plt.barh(range(len(indices)), gbfi[indices], color='b', align='center')
plt.yticks(range(len(indices)), [dataall.columns[i] for i in indices])

plt.show()



#%%

#k-fold cross-validation using MSLE
cval = cross_validate(rfModel, dataall,rphall, scoring=metrics.make_scorer(metrics.mean_squared_log_error), cv=10)

print(cval['test_score'])




#%%

#Neural network


yLabels = dfb['ridesperhour']


model = Sequential()
model.add(Dense(32, input_dim=11, activation='relu'))     #CHANGE INPUT_DIM to 25 if using census data or 11 if not (also change if dropping features)
model.add(Dense(64, activation='relu'))
model.add(k.layers.Dense(1))   #output predicted rides per hour

#Adam optimizer is used for preliminary results. Other optimizers may be more stable/robust, perhaps less efficient
#Use Keras MSLE metric
model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError()) #I think this is MeanSquaredError function

#Adjust number of epochs (iterations over set), validation split, other arguments, and verbosity
#Change input as needed
history = model.fit(dataallwo, yLabels, validation_split=0.2, epochs=200,verbose=0)
#Change dataallwo to dataall to include census data



print("NN Average RMSE: ",np.average(history.history['loss']))


print("NN Average Normalized RMSE: ",np.average(history.history['loss'])/(max(yLabels)-min(yLabels)))



#%%
#Model evaluation

evaltest = model.evaluate(x,y,batch_size=1)
#print('Accuracy: %.2f' % (accuracy*100))
print(evaltest)


#%%

#Plot loss over epochs

plt.plot(history.history['val_loss'])
plt.show()



#%%

#Not used in preliminary results

permut = PermutationImportance(model, scoring="accuracy").fit(testdata,yLabels)
eli5.show_weights(permut, feature_names = dataall.columns.tolist())


