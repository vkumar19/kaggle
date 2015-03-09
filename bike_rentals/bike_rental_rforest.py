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

rental_df['date'] = rental_df.datetime.apply(lambda x: x.date())
test_df['date'] = test_df.datetime.apply(lambda x: x.date())

grp_date = rental_df.groupby('date')
grp_test_date = test_df.groupby('date')

def avg_temp(x):
    return grp_date.get_group(x)['temp'].mean()
def test_avg_temp(x):
    return grp_test_date.get_group(x)['temp'].mean()

rental_df['avg_temp'] = rental_df['date'].apply(lambda x: avg_temp(x))
test_df['avg_temp'] = test_df['date'].apply(lambda x: test_avg_temp(x))

rental_df['hour'] = rental_df.datetime.apply(lambda x: x.hour+1)
test_df['hour'] = test_df.datetime.apply(lambda x: x.hour+1)

rental_df['month'] = rental_df.datetime.apply(lambda x: x.month)
test_df['month'] = test_df.datetime.apply(lambda x: x.month)

rental_df['weekend'] = rental_df.datetime.apply(lambda x: weekend(x))
test_df['weekend'] = rental_df.datetime.apply(lambda x: weekend(x))

rental_df['2011'] = rental_df.datetime.apply(lambda x: year_2011(x))
test_df['2011'] = test_df.datetime.apply(lambda x: year_2011(x))

ml_df = rental_df.drop(['datetime', 'casual', 'windspeed', 'count', 'temp', 'atemp', 'registered', 'humidity', 'month', 'date', 'weekend', 'holiday'], axis = 1)

ml_df_test = test_df.drop(['datetime','windspeed', 'humidity', 'month', 'date', 'temp', 'atemp', 'weekend', 'holiday'], axis = 1)
print ml_df.info()
print ml_df_test.info()

ml_df = np.array(ml_df)
ml_df_test = np.array(ml_df_test)

Y = np.array(rental_df['count'])

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, oob_score=True)
rf.fit(ml_df, Y)

temp = np.log(rf.predict(ml_df)+1) - np.log(Y+1)
temp = temp*temp
RMSLE = np.mean(temp)
RMSLE = RMSLE**0.5
count = rf.predict(ml_df_test)

df_submission = pd.DataFrame(count, test_df.datetime, columns = ['count'])
pd.DataFrame.to_csv(df_submission ,'randomforest_predict.csv')

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(rf.predict(ml_df))
plt.plot(Y)
plt.show()

from sklearn import cross_validation
from sklearn.metrics import make_scorer

def my_custom_loss_func(ground_truth, predictions):
  diff = np.log(ground_truth+1) - np.log(predictions+1)
  diff = diff*diff
  diff = np.mean(diff)
  return diff

my_custom_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)

scores = cross_validation.cross_val_score(rf, ml_df, Y, cv=5, scoring=my_custom_scorer)
