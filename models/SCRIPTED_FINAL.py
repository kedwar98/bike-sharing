# %%

# Use "Run cell" to run an individual cell
# You can use "#%%" to create a cell
# Code is set up in 3 main sections, Baseline Models without census data, total with census data, holdout city models with and without census data and with selected features

# Run this cell for import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math

# %% Mean squared logarithmic error
from sklearn.metrics import mean_squared_log_error

# %%
# Read in all data to be used

totaldata = pd.read_csv('WEATHER-CENSUS/totaldata.csv')
bosdata = pd.read_csv('WEATHER-CENSUS/Bostondata.csv')
nobosdata = pd.read_csv('WEATHER-CENSUS/NO_BOSTON.csv')
nycdata = pd.read_csv('WEATHER-CENSUS/NYCdata.csv')
nonycdata = pd.read_csv('WEATHER-CENSUS/NO_NYC.csv')
phildata = pd.read_csv('WEATHER-CENSUS/Philadelphiadata.csv')
nophildata = pd.read_csv('WEATHER-CENSUS/NO_Philadelphia.csv')
chicdata = pd.read_csv('WEATHER-CENSUS/Chicagodata.csv')
nochicdata = pd.read_csv('WEATHER-CENSUS/NO_Chicago.csv')
ladata = pd.read_csv('WEATHER-CENSUS/LAdata.csv')
noladata = pd.read_csv('WEATHER-CENSUS/NO_LA.csv')
dcdata = pd.read_csv('WEATHER-CENSUS/DCdata.csv')
nodcdata = pd.read_csv('WEATHER-CENSUS/NO_DC.csv')
cities = [ladata, bosdata, nycdata, chicdata, dcdata, phildata, totaldata]
names = ['LA', 'BOSTON', 'NYC', 'CHICAGO', 'DC', 'PHILADELPHIA', 'total']

# # %%Find Baseline Data for Each City and Combined W/O Census Data
n = 0
for a in cities:
    dfb = a.copy()
    dfb = dfb.sample(frac=1).reset_index(drop=True)

    # Establish 80-20 train-test splits for each city of 8760 rows

    traindata = dfb.copy()
    traindata = traindata[0:int(0.8 * len(traindata))]
    trainlabels = traindata['ridesperhour']
    testdata = dfb.copy()
    testdata = testdata[int(0.8 * len(testdata)):]
    testlabels = testdata['ridesperhour']

    # drop rides per hour and unnamed columns
    traindata.drop('ridesperhour', axis=1, inplace=True)
    testdata.drop('ridesperhour', axis=1, inplace=True)
    traindata.drop(traindata.columns[traindata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    testdata.drop(testdata.columns[testdata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    # drop census data
    traindatawo = traindata.copy()
    traindatawo = traindatawo.iloc[:, 0:11]
    testdatawo = testdata.copy()
    testdatawo = testdatawo.iloc[:, 0:11]

    # Random Forest Model from sklearn
    rfModel = RandomForestRegressor(n_estimators=1000, verbose=0)
    rfModel.fit(traindatawo, trainlabels)
    # Training error using training data
    preds = rfModel.predict(X=traindatawo)
    # set any negative predictions to 0 rph
    for i in range(len(preds)):
        if preds[i] < 0:
            preds[i] = 0
    print(names[n], "RF Training RMSLE WITHOUT CENSUS DATA:", math.sqrt(mean_squared_log_error(trainlabels, preds)))

    # Test model predictions using 20% held out testing data
    testpreds = rfModel.predict(X=testdatawo)
    # set any negative predictions to 0 rph
    for i in range(len(testpreds)):
        if testpreds[i] < 0:
            testpreds[i] = 0
    print(names[n], "RF Testing RMSLE WITHOUT CENSUS DATA:", math.sqrt(mean_squared_log_error(testlabels, testpreds)))
    # #Plot RF feature importance
    # rffi=rfModel.feature_importances_
    # indices = np.argsort(rffi)[::-1]
    # plt.barh(range(len(indices)), rffi[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), [testdata.columnswo[i] for i in indices])
    #
    # plt.show()

    # Gradient Boosting Model from sklearn

    # Here you can change the number of estimators, the L1 regularization coefficient alpha, and other arguments, as well as verbosity
    gbModel = GradientBoostingRegressor(n_estimators=1000, alpha=0.01, verbose=0)
    gbModel.fit(traindatawo, trainlabels)
    preds = gbModel.predict(X=traindatawo)
    # set any negative predictions to 0 rph
    for i in range(len(preds)):
        if preds[i] < 0:
            preds[i] = 0
    print(names[n], "GB Training RMSLE WITHOUT CENSUS DATA:", math.sqrt(mean_squared_log_error(trainlabels, preds)))
    # Test model predictions using 20% held out testing data
    testpreds = gbModel.predict(X=testdatawo)
    # set any negative predictions to 0 rph
    for i in range(len(testpreds)):
        if testpreds[i] < 0:
            testpreds[i] = 0
    print(names[n], "GB Testing RMSLE WITHOUT CENSUS DATA:", math.sqrt(mean_squared_log_error(testlabels, testpreds)))
    # #Plot GB feature impoptance
    # gbfi=gbModel.feature_importances_
    # indices = np.argsort(gbfi)[::-1]
    #
    # plt.barh(range(len(indices)), gbfi[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), [traindatawo.columns[i] for i in indices])
    #
    # plt.show()

    # NN
    model = Sequential()
    model.add(k.layers.BatchNormalization())

    model.add(Dense(128, input_dim=11, activation='relu'))
    model.add(k.layers.BatchNormalization())
    # model.add(k.layers.Activation(tf.keras.activations.selu))

    model.add(Dense(256, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))

    model.add(Dense(128, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))

    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())

    history = model.fit(traindatawo, trainlabels, validation_split=0.2, epochs=200, verbose=2)
    test_error = model.evaluate(testdatawo, testlabels)
    test_error = math.sqrt(test_error)
    print(names[n], 'NN RMSLE test loss, test acc WITHOUT CENSUS DATA:', test_error)
    print(names[n], "NN Average RMSE over last 50 points WITHOUT CENSUS DATA: ",
          np.average(history.history['val_loss'][-50:]))
    # plt.plot(history.history['val_loss'])
    # plt.show()
    n += 1

#%%Find Combined W/Census Data
#redefine traindata/labels and testdata/labels
dfb = totaldata.copy()
dfb = dfb.sample(frac=1).reset_index(drop=True)

traindata = dfb.copy()
traindata = traindata[0:int(0.8 * len(traindata))]
trainlabels = traindata['ridesperhour']
testdata = dfb.copy()
testdata = testdata[int(0.8 *len(testdata)):]
testlabels = testdata['ridesperhour']
# drop rides per hour and unnamed columns
traindata.drop('ridesperhour', axis=1, inplace=True)
testdata.drop('ridesperhour', axis=1, inplace=True)
traindata.drop(traindata.columns[traindata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
testdata.drop(testdata.columns[testdata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Random Forest Model from sklearn
rfModel = RandomForestRegressor(n_estimators=1000, verbose=0)
rfModel.fit(traindata, trainlabels)
# Training error using training data
preds = rfModel.predict(X=traindata)
# set any negative predictions to 0 rph
for i in range(len(preds)):
    if preds[i] < 0:
        preds[i] = 0
print('total', "RF Training RMSLE WITH CENSUS DATA:", math.sqrt(mean_squared_log_error(trainlabels, preds)))
rf_err = math.sqrt(mean_squared_log_error(trainlabels, preds))
testpreds = rfModel.predict(X=testdata)


# set any negative predictions to 0 rph
for i in range(len(testpreds)):
    if testpreds[i] < 0:
        testpreds[i] = 0
rf_testpreds = testpreds

#Plot predictions vs True Labels
plt.scatter(testpreds,testlabels)
plt.show()
print('total', "RF Testing RMSLE WITH CENSUS DATA:", math.sqrt(mean_squared_log_error(testlabels, testpreds)))

#Plot RF feature importance
rffi=rfModel.feature_importances_
indices = np.argsort(rffi)[::-1]
plt.barh(range(len(indices)), rffi[indices], color='b', align='center')
plt.yticks(range(len(indices)), [testdata.columns[i] for i in indices])

plt.show()

# Gradient Boosting Model from sklearn combined with Census Data
# Here you can change the number of estimators, the L1 regularization coefficient alpha, and other arguments, as well as verbosity
gbModel = GradientBoostingRegressor(n_estimators=1000, alpha=0.01, verbose=0)
gbModel.fit(traindata, trainlabels)
preds = gbModel.predict(X=traindata)
# set any negative predictions to 0 rph
for i in range(len(preds)):
    if preds[i] < 0:
        preds[i] = 0
print('total', "GB Training RMSLE WITH CENSUS DATA:", math.sqrt(mean_squared_log_error(trainlabels, preds)))
gb_err = math.sqrt(mean_squared_log_error(trainlabels, preds))

# Test model predictions using 20% held out testing data
testpreds = gbModel.predict(X=testdata)
# set any negative predictions to 0 rph
for i in range(len(testpreds)):
    if testpreds[i] < 0:
        testpreds[i] = 0
print('total', "GB Testing RMSLE WITH CENSUS DATA:", math.sqrt(mean_squared_log_error(testlabels, testpreds)))
gb_testpreds=testpreds
plt.scatter(testpreds,testlabels)
plt.show()
# Plot GB feature impoptance
gbfi=gbModel.feature_importances_
indices = np.argsort(gbfi)[::-1]

plt.barh(range(len(indices)), gbfi[indices], color='b', align='center')
plt.yticks(range(len(indices)), [traindata.columns[i] for i in indices])

plt.show()

# NN Combined With Census Data
model = Sequential()
model.add(k.layers.BatchNormalization())

model.add(Dense(128, input_dim=25, activation='relu'))
model.add(k.layers.BatchNormalization())
# model.add(k.layers.Activation(tf.keras.activations.selu))

model.add(Dense(256, activation='relu'))
model.add(k.layers.BatchNormalization())
#model.add(k.layers.Activation(tf.keras.activations.selu))

model.add(Dense(128, activation='relu'))
model.add(k.layers.BatchNormalization())
#model.add(k.layers.Activation(tf.keras.activations.selu))

model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())

history = model.fit(traindata, trainlabels, validation_split=0.2, epochs=200, verbose=0)
test_error = model.evaluate(testdata, testlabels)
test_error = math.sqrt(test_error)
print('total', 'NN RMSLE test loss, test acc WITH CENSUS DATA:', test_error)
print('total', "NN Average RMSE over last 50 points WITH CENSUS DATA: ", np.average(history.history['val_loss'][-50:]))
NN_err = np.average(history.history['val_loss'][-50:])
NN_err = math.sqrt(NN_err)
NN_testpreds = np.ndarray.flatten(np.array(model.predict(testdata)))



NN_weight = (rf_err+gb_err)/(rf_err+NN_err+gb_err)
print('NN_weight',NN_weight)
rf_weight = (NN_err+gb_err)/(rf_err+NN_err+gb_err)
print('rf_weight',rf_weight)
gb_weight = (rf_err+NN_err)/(rf_err+NN_err+gb_err)
print('gb_weight',gb_weight)

ens_preds_equal = (1/3)*NN_testpreds+ (1/3)*rf_testpreds + (1/3)*gb_testpreds
ens_preds_w = (NN_weight*NN_testpreds+rf_weight*rf_testpreds+gb_weight*gb_testpreds)/2

print('ens_error_equal', math.sqrt(mean_squared_log_error(testlabels, ens_preds_equal)))
print('ens_error_weighted',math.sqrt(mean_squared_log_error(testlabels, ens_preds_w)))

#Plot NN history of loss
plt.plot(history.history['val_loss'])
plt.show()

plt.scatter(NN_testpreds,testlabels)
plt.show()
#
# # %%Test Holdout Data by holding out each city and seeing how well the model does at predicting that city with and without census data
# nocities = [noladata, nobosdata, nonycdata, nochicdata, nodcdata, nophildata]
# n = 0
# for a in nocities:
#     dfnew = a.copy()
#     # shuffle training data
#     dfnew = dfnew.sample(frac=1).reset_index(drop=True)
#
#     # Set training data equal to the data w/out a city and testing data equal to the city alone data
#
#     traindata = dfnew.copy()
#     trainlabels = traindata['ridesperhour']
#     testdata = cities[n].copy()
#     testlabels = testdata['ridesperhour']
#
#     # drop rides per hour and unnamed columns
#     traindata.drop('ridesperhour', axis=1, inplace=True)
#     testdata.drop('ridesperhour', axis=1, inplace=True)
#     traindata.drop(traindata.columns[traindata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
#     testdata.drop(testdata.columns[testdata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
#
#      # drop census data
#     traindatawo = traindata.copy()
#     traindatawo = traindatawo.iloc[:, 0:11]
#     testdatawo = testdata.copy()
#     testdatawo = testdatawo.iloc[:, 0:11]
#
#     # drop non-relevant data
#     traindatasel = traindata.copy()
#     traindatasel.drop('HourlyDewPointTemperature',axis=1,inplace=True)
#     traindatasel.drop('HourlyPrecipitation',axis=1,inplace=True)
#     traindatasel.drop('HourlyVisibility',axis=1,inplace=True)
#     traindatasel.drop('HourlyWetBulbTemperature',axis=1,inplace=True)
#     traindatasel.drop('HourlyWindGustSpeed',axis=1,inplace=True)
#     traindatasel.drop('HourlyWindSpeed',axis=1,inplace=True)
#     traindatasel.drop('Total_population_SUMMARY_INDICATORS_Sex_ratio_(males_per_100_females)',axis=1,inplace=True)
#     traindatasel.drop('Estimate_Percent_Total_population_15_to_44_years',axis=1,inplace=True)
#     traindatasel.drop('MEDIAN_INCOME_PER_100k',axis=1,inplace=True)
#     traindatasel.drop('RATIO_OF_NON-FAMILY_TO_FAMILY_HOUSEHOLDS',axis=1,inplace=True)
#     traindatasel.drop('RATIO_OF_WOKERS_TO_PEOPLE_(OVER_16)',axis=1,inplace=True)
#     traindatasel.drop('Workers_16_years_and_over_Walked',axis=1,inplace=True)
#     traindatasel.drop('Workers_16_years_and_over_Bicycle',axis=1,inplace=True)
#     traindatasel.drop('Workers_16_years_and_over_Taxicab,_motorcycle,_or_other_means',axis=1,inplace=True)
#     traindatasel.drop('month',axis=1,inplace=True)
#     traindatasel.drop('day',axis=1,inplace=True)
#
#     testdatasel = testdata.copy()
#     testdatasel.drop('HourlyDewPointTemperature',axis=1,inplace=True)
#     testdatasel.drop('HourlyPrecipitation',axis=1,inplace=True)
#     testdatasel.drop('HourlyVisibility',axis=1,inplace=True)
#     testdatasel.drop('HourlyWetBulbTemperature',axis=1,inplace=True)
#     testdatasel.drop('HourlyWindGustSpeed',axis=1,inplace=True)
#     testdatasel.drop('HourlyWindSpeed',axis=1,inplace=True)
#     testdatasel.drop('Total_population_SUMMARY_INDICATORS_Sex_ratio_(males_per_100_females)',axis=1,inplace=True)
#     testdatasel.drop('Estimate_Percent_Total_population_15_to_44_years',axis=1,inplace=True)
#     testdatasel.drop('MEDIAN_INCOME_PER_100k',axis=1,inplace=True)
#     testdatasel.drop('RATIO_OF_NON-FAMILY_TO_FAMILY_HOUSEHOLDS',axis=1,inplace=True)
#     testdatasel.drop('RATIO_OF_WOKERS_TO_PEOPLE_(OVER_16)',axis=1,inplace=True)
#     testdatasel.drop('Workers_16_years_and_over_Walked',axis=1,inplace=True)
#     testdatasel.drop('Workers_16_years_and_over_Bicycle',axis=1,inplace=True)
#     testdatasel.drop('Workers_16_years_and_over_Taxicab,_motorcycle,_or_other_means',axis=1,inplace=True)
#     testdatasel.drop('month',axis=1,inplace=True)
#     testdatasel.drop('day',axis=1,inplace=True)
#
#     # Random Forest Model from sklearn W/O Census data with holdout city
#     rfModel = RandomForestRegressor(n_estimators=1000, verbose=0)
#     rfModel.fit(traindatawo, trainlabels)
#     # Training error using training data
#     preds = rfModel.predict(X=traindatawo)
#
#     # set any negative predictions to 0 rph
#     for i in range(len(preds)):
#         if preds[i] < 0:
#             preds[i] = 0
#     print("Holding Out", names[n], "RF Training RMSLE WITHOUT CENSUS DATA:",
#           math.sqrt(mean_squared_log_error(trainlabels, preds)))
#
#     # Test model predictions using held out city
#     testpreds = rfModel.predict(X=testdatawo)
#     # set any negative predictions to 0 rph
#     for i in range(len(testpreds)):
#         if testpreds[i] < 0:
#             testpreds[i] = 0
#     print("Holding Out", names[n], "RF Testing RMSLE WITHOUT CENSUS DATA:",
#           math.sqrt(mean_squared_log_error(testlabels, testpreds)))
#     rf_wo_testpreds = testpreds.copy()
#
#     # #Plot RF feature importance
#     # rffi=rfModel.feature_importances_
#     # indices = np.argsort(rffi)[::-1]
#     # plt.barh(range(len(indices)), rffi[indices], color='b', align='center')
#     # plt.yticks(range(len(indices)), [testdata.columnswo[i] for i in indices])
#     #
#     # plt.show()
#
#     # Random Forest Model from sklearn with census data with holdout city
#     rfModel = RandomForestRegressor(n_estimators=1000, verbose=0)
#     rfModel.fit(traindata, trainlabels)
#     # Training error using training data
#     preds = rfModel.predict(X=traindata)
#     # set any negative predictions to 0 rph
#     for i in range(len(preds)):
#         if preds[i] < 0:
#             preds[i] = 0
#     print("Holding Out", names[n], "RF Training RMSLE WITH CENSUS DATA:",
#           math.sqrt(mean_squared_log_error(trainlabels, preds)))
#
#     # Test model predictions using held out city
#     testpreds = rfModel.predict(X=testdata)
#     # set any negative predictions to 0 rph
#     for i in range(len(testpreds)):
#         if testpreds[i] < 0:
#             testpreds[i] = 0
#     print("Holding Out", names[n], "RF Testing RMSLE WITH CENSUS DATA :",
#           math.sqrt(mean_squared_log_error(testlabels, testpreds)))
#     rf_testpreds = testpreds.copy()
#
#     # Random Forest Model from sklearn with only selected features
#     rfModel = RandomForestRegressor(n_estimators=1000, verbose=0)
#     rfModel.fit(traindatasel, trainlabels)
#     # Training error using training data
#     preds = rfModel.predict(X=traindatasel)
#     # set any negative predictions to 0 rph
#     for i in range(len(preds)):
#         if preds[i] < 0:
#             preds[i] = 0
#     print("Holding Out", names[n], "RF Training RMSLE WITH SELECTED DATA:",
#           math.sqrt(mean_squared_log_error(trainlabels, preds)))
#
#     # Test model predictions using held out city
#     testpreds = rfModel.predict(X=testdatasel)
#     # set any negative predictions to 0 rph
#     for i in range(len(testpreds)):
#         if testpreds[i] < 0:
#             testpreds[i] = 0
#     print("Holding Out", names[n], "RF Testing RMSLE WITH SELECTED DATA :",
#           math.sqrt(mean_squared_log_error(testlabels, testpreds)))
#     rf_sel_testpreds = testpreds.copy()
#     #Plot RF feature importance
#     rffi=rfModel.feature_importances_
#     indices = np.argsort(rffi)[::-1]
#     plt.barh(range(len(indices)), rffi[indices], color='b', align='center')
#     plt.yticks(range(len(indices)), [testdata.columnswo[i] for i in indices])
#     plt.show()
#
#     # Gradient Boosting Model from sklearn w/out census data
#     # Here you can change the number of estimators, the L1 regularization coefficient alpha, and other arguments, as well as verbosity
#     gbModel = GradientBoostingRegressor(n_estimators=1000, alpha=0.01, verbose=0)
#     gbModel.fit(traindatawo, trainlabels)
#     preds = gbModel.predict(X=traindatawo)
#     # set any negative predictions to 0 rph
#     for i in range(len(preds)):
#         if preds[i] < 0:
#             preds[i] = 0
#     print("Holding Out", names[n], "GB Training RMSLE WITHOUT CENSUS DATA:",
#           math.sqrt(mean_squared_log_error(trainlabels, preds)))
#     # Test model predictions using held out city
#     testpreds = gbModel.predict(X=testdatawo)
#     # set any negative predictions to 0 rph
#     for i in range(len(testpreds)):
#         if testpreds[i] < 0:
#             testpreds[i] = 0
#     print("Holding Out", names[n], "GB Testing RMSLE WITHOUT CENSUS DATA:",
#           math.sqrt(mean_squared_log_error(testlabels, testpreds)))
#     gb_wo_testpreds = testpreds.copy()
#     #Plot GB feature impoptance
#     gbfi=gbModel.feature_importances_
#     indices = np.argsort(gbfi)[::-1]
#
#     plt.barh(range(len(indices)), gbfi[indices], color='b', align='center')
#     plt.yticks(range(len(indices)), [traindatawo.columns[i] for i in indices])
#
#     plt.show()
#
#     # %%Gradient Boosting Model from sklearn w census data
#     # Here you can change the number of estimators, the L1 regularization coefficient alpha, and other arguments, as well as verbosity
#     gbModel = GradientBoostingRegressor(n_estimators=1000, alpha=0.01, verbose=0)
#     gbModel.fit(traindata, trainlabels)
#     preds = gbModel.predict(X=traindata)
#     # set any negative predictions to 0 rph
#     for i in range(len(preds)):
#         if preds[i] < 0:
#             preds[i] = 0
#     print("Holding Out", names[n], "GB Training RMSLE WITH CENSUS DATA:",
#           math.sqrt(mean_squared_log_error(trainlabels, preds)))
#     # Test model predictions using held out city
#     testpreds = gbModel.predict(X=testdata)
#     # set any negative predictions to 0 rph
#     for i in range(len(testpreds)):
#         if testpreds[i] < 0:
#             testpreds[i] = 0
#     print("Holding Out", names[n], "GB Testing RMSLE WITH CENSUS DATA:",
#           math.sqrt(mean_squared_log_error(testlabels, testpreds)))
#     gb_testpreds = testpreds.copy()
#     #Plot GB feature impoptance
#     gbfi=gbModel.feature_importances_
#     indices = np.argsort(gbfi)[::-1]
#
#     plt.barh(range(len(indices)), gbfi[indices], color='b', align='center')
#     plt.yticks(range(len(indices)), [traindatawo.columns[i] for i in indices])
#
#     plt.show()
#
#     # %%Gradient Boosting Model from sklearn w selected features
#     # Here you can change the number of estimators, the L1 regularization coefficient alpha, and other arguments, as well as verbosity
#     gbModel = GradientBoostingRegressor(n_estimators=1000, alpha=0.01, verbose=0)
#     gbModel.fit(traindatasel, trainlabels)
#     preds = gbModel.predict(X=traindatasel)
#
#     # set any negative predictions to 0 rph
#     for i in range(len(preds)):
#         if preds[i] < 0:
#             preds[i] = 0
#     print("Holding Out", names[n], "GB Training RMSLE WITH SELECTED DATA:",
#           math.sqrt(mean_squared_log_error(trainlabels, preds)))
#     # Test model predictions using held out city
#     testpreds = gbModel.predict(X=testdatasel)
#
#     # set any negative predictions to 0 rph
#     for i in range(len(testpreds)):
#         if testpreds[i] < 0:
#             testpreds[i] = 0
#     print("Holding Out", names[n], "GB Testing RMSLE WITH SELECTED DATA:",
#           math.sqrt(mean_squared_log_error(testlabels, testpreds)))
#     gb_testpreds_sel = testpreds.copy()
#
#     # NN without census data
#     model = Sequential()
#     model.add(k.layers.BatchNormalization())
#
#     model.add(Dense(128, input_dim=11, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#
#
#     model.add(Dense(256, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#
#
#     model.add(Dense(128, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#
#
#     model.add(Dense(1, activation='relu'))
#     model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
#
#     history = model.fit(traindatawo, trainlabels, validation_split=0.2, epochs=200, verbose=0)
#     test_error = model.evaluate(testdatawo, testlabels)
#     test_error = math.sqrt(test_error)
#
#     print("Holding Out", names[n], 'NN RMSLE without census data test loss, test acc', test_error)
#     print("Holding Out", names[n], "NN Average RMSLE without census data over last 50 points: ",
#           np.average(history.history['val_loss'][-50:]))
#     NN_wo_testpreds = np.ndarray.flatten(np.array(model.predict(testdatawo)))
#
#
#
#     # plt.plot(history.history['val_loss'])
#     # plt.show()
#
#     # NN with census data
#     model = Sequential()
#     model.add(k.layers.BatchNormalization())
#
#     model.add(Dense(128, input_dim=25, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#     # model.add(k.layers.Activation(tf.keras.activations.selu))
#
#     model.add(Dense(256, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#     #model.add(k.layers.Activation(tf.keras.activations.selu))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#     #model.add(k.layers.Activation(tf.keras.activations.selu))
#
#     model.add(Dense(1, activation='relu'))
#     model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
#
#     history = model.fit(traindata, trainlabels, validation_split=0.2, epochs=200, verbose=0)
#     test_error = model.evaluate(testdata, testlabels)
#     test_error = math.sqrt(test_error)
#     print("Holding Out", names[n], 'NN RMSLE with census data test loss, test acc', test_error)
#     print("Holding Out", names[n], "NN Average RMSLE with census data over last 50 points: ",
#           np.average(history.history['val_loss'][-50:]))
#     NN_testpreds = np.ndarray.flatten(np.array(model.predict(testdata)))
#
#     # NN with selected data
#     model = Sequential()
#     model.add(k.layers.BatchNormalization())
#
#     model.add(Dense(128, input_dim=9, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#     # model.add(k.layers.Activation(tf.keras.activations.selu))
#
#     model.add(Dense(256, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#     #model.add(k.layers.Activation(tf.keras.activations.selu))
#
#     model.add(Dense(128, activation='relu'))
#     model.add(k.layers.BatchNormalization())
#     #model.add(k.layers.Activation(tf.keras.activations.selu))
#
#     model.add(Dense(1, activation='relu'))
#     model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
#
#     history = model.fit(traindatasel, trainlabels, validation_split=0.2, epochs=200, verbose=0)
#     test_error = model.evaluate(testdatasel, testlabels)
#     test_error = math.sqrt(test_error)
#     print("Holding Out", names[n], 'NN RMSLE with selected data test loss, test acc', test_error)
#     print("Holding Out", names[n], "NN Average RMSLE with selected data over last 50 points: ",
#           np.average(history.history['val_loss'][-50:]))
#     NN_testpreds_sel = np.ndarray.flatten(np.array(model.predict(testdatasel)))
#
#     wo_ensemble = (NN_wo_testpreds+rf_wo_testpreds+gb_wo_testpreds)/3
#     wo_error = math.sqrt(mean_squared_log_error(testlabels, wo_ensemble))
#     print("Holding Out",names[n],"Ensemble without census data Error",wo_error)
#     w_ensemble = (NN_testpreds+rf_testpreds+gb_testpreds)/3
#     w_error = math.sqrt(mean_squared_log_error(testlabels,w_ensemble))
#     print("Holding Out",names[n],"Ensemble with census data Error",w_error)
#     sel_ensemble = (NN_testpreds_sel+rf_sel_testpreds+gb_testpreds_sel)/3
#     sel_error = math.sqrt(mean_squared_log_error(testlabels,sel_ensemble))
#     print("Holding Out",names[n],"Ensemble with selected data Error",sel_error)
#     n+=1
