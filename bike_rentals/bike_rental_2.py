import pandas as pd
import numpy as np

def weekend(x):
  if x.weekday() == 5 or x.weekday() == 6:
    return 1
  else:
    return 0

def year_2011(x):
  if x.year == 2011:
    return 1
  else:
    return 0
  
rental_df = pd.read_csv("train.csv", parse_dates = True)
test_df = pd.read_csv("test.csv", parse_dates = True)

rental_df['datetime'] = pd.to_datetime(rental_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

rental_df['hour'] = rental_df.datetime.apply(lambda x: x.hour+1)
test_df['hour'] = test_df.datetime.apply(lambda x: x.hour+1)

rental_df['weekday'] = rental_df.datetime.apply(lambda x: x.weekday()+1)
test_df['weekday'] = test_df.datetime.apply(lambda x: x.weekday()+1)

rental_df['month'] = rental_df.datetime.apply(lambda x: x.month)
test_df['month'] = test_df.datetime.apply(lambda x: x.month)

rental_df['weekend'] = rental_df.datetime.apply(lambda x: weekend(x))
test_df['weekend'] = rental_df.datetime.apply(lambda x: weekend(x))

rental_df['2011'] = rental_df.datetime.apply(lambda x: year_2011(x))
test_df['2011'] = test_df.datetime.apply(lambda x: year_2011(x))

dummy_season = pd.get_dummies(rental_df['season'], prefix='season')
rental_df = rental_df.join(dummy_season.ix[:,'season_2':])
dumtest_season = pd.get_dummies(test_df['season'], prefix='season')
test_df = test_df.join(dumtest_season.ix[:,'season_2':])


dummy_weather = pd.get_dummies(rental_df['weather'], prefix='weather')
rental_df = rental_df.join(dummy_weather.ix[:,'weather_2':])
dumtest_weather = pd.get_dummies(test_df['weather'], prefix='weather')
test_df = test_df.join(dumtest_weather.ix[:,'weather_2':])

dummy_hour = pd.get_dummies(rental_df['hour'], prefix='hour')
rental_df = rental_df.join(dummy_hour.ix[:,'hour_2':])
dumtest_hour = pd.get_dummies(test_df['hour'], prefix='hour')
test_df = test_df.join(dumtest_hour.ix[:,'hour_2':])

dummy_month = pd.get_dummies(rental_df['month'], prefix='month')
rental_df = rental_df.join(dummy_month.ix[:,'month_2':])
dumtest_month = pd.get_dummies(test_df['month'], prefix='month')
test_df = test_df.join(dumtest_month.ix[:,'month_2':])


ml_df = rental_df.drop(['datetime', 'casual', 'count', 'season', 'hour', 'weather', 'weekday', 'month', 'registered'], axis = 1)

ml_df_test = test_df.drop(['datetime','season', 'hour', 'weather', 'weekday', 'month'], axis = 1)
ml_df = np.array(ml_df)
ml_df_test = np.array(ml_df_test)
#enc = preprocessing.OneHotEncoder(categorical_features = [0, 3, 8, 9])
#enc.fit(ml_df)

#temp = enc.transform(ml_df).toarray()
Y = np.array(rental_df['count'])

from sklearn import linear_model
num = 0.01
clf = linear_model.Ridge(alpha = num, fit_intercept=True, normalize=True, max_iter = 10000)
clf.fit(ml_df, Y)
temp = np.log(abs(clf.predict(ml_df))+1) - np.log(Y+1)
#temp = abs(clf.predict(ml_df)) - Y
temp = temp*temp
RMSLE = np.mean(temp)
RMSLE = RMSLE**0.5
count = abs(clf.predict(ml_df_test))

df_submission = pd.DataFrame(count, test_df.datetime, columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'randomforest_predict.csv')

#dummy_weekday = pd.get_dummies(rental_df['weekday'], prefix='weekday')
#rental_df = rental_df.join(dummy_weekday.ix[:,'weekday_2':])
#dumtest_weekday = pd.get_dummies(test_df['weekday'], prefix='weekday')
#test_df = test_df.join(dumtest_weekday.ix[:,'weekday_2':])

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(clf.predict(ml_df))
plt.plot(Y)
plt.show()
