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
def avg_windspeed(x):
    return grp_date.get_group(x)['windspeed'].mean()
def test_avg_windspeed(x):
    return grp_test_date.get_group(x)['windspeed'].mean()
def avg_humidity(x):
    return grp_date.get_group(x)['humidity'].mean()
def test_avg_humidity(x):
    return grp_test_date.get_group(x)['humidity'].mean()

rental_df['hour'] = rental_df.datetime.apply(lambda x: x.hour)
test_df['hour'] = test_df.datetime.apply(lambda x: x.hour)

rental_df['avg_temp'] = rental_df['date'].apply(lambda x: avg_temp(x))
test_df['avg_temp'] = test_df['date'].apply(lambda x: test_avg_temp(x))

rental_df['avg_wspeed'] = rental_df['date'].apply(lambda x: avg_windspeed(x))
test_df['avg_wspeed'] = test_df['date'].apply(lambda x: test_avg_windspeed(x))

rental_df['avg_humidity'] = rental_df['date'].apply(lambda x: avg_humidity(x))
test_df['avg_humidity'] = test_df['date'].apply(lambda x: test_avg_humidity(x))

rental_df['2011'] = rental_df.datetime.apply(lambda x: year_2011(x))
test_df['2011'] = test_df.datetime.apply(lambda x: year_2011(x))

df_grouped = rental_df.groupby(['hour'])
df_grpd_test = test_df.groupby(['hour'])

hr_cnt_holiday = [0 for i in range(0,24)]
hr_cnt_workingday = [0 for i in range(0,24)]

for name, group in df_grouped:
    norm_hr_holiday = 0
    norm_hr_workingday = 0
    for i in range(0,len(group.workingday)):
        if group.workingday.values[i] == 0:
            norm_hr_holiday += 1
            hr_cnt_holiday[name] += group['count'].values[i]
        else:
            norm_hr_workingday += 1
            hr_cnt_workingday[name] += group['count'].values[i]
    hr_cnt_holiday[name] /= norm_hr_holiday
    hr_cnt_workingday[name] /= norm_hr_workingday 

rental_df['count_avg_1'] = rental_df[rental_df['workingday'] == 0]['hour'].apply(lambda x: hr_cnt_holiday[x])
rental_df['count_avg_2'] = rental_df[rental_df['workingday'] == 1]['hour'].apply(lambda x: hr_cnt_workingday[x])

rental_df['count_avg'] = rental_df['count_avg_1'].fillna(0) + rental_df['count_avg_2'].fillna(0)

test_df['count_avg_1'] = test_df[test_df['workingday'] == 0]['hour'].apply(lambda x: 
hr_cnt_holiday[x])

test_df['count_avg_2'] = test_df[test_df['workingday'] == 1]['hour'].apply(lambda x: hr_cnt_workingday[x])

test_df['count_avg'] = test_df['count_avg_1'].fillna(0) + test_df['count_avg_2'].fillna(0)

ml_df = rental_df.drop(['datetime', 'casual', 'count', 'registered', 'date', 'holiday', 'count_avg_1', 'count_avg_2'], axis = 1)

ml_df_test = test_df.drop(['datetime', 'date', 'holiday','count_avg_1', 'count_avg_2'], axis = 1)

print ml_df.info()
print ml_df_test.info()

ml_df = np.array(ml_df)
ml_df_test = np.array(ml_df_test)
#enc = preprocessing.OneHotEncoder(categorical_features = [0, 3, 8, 9])
#enc.fit(ml_df)

#temp = enc.transform(ml_df).toarray()
Y = np.array(rental_df['count'])

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, oob_score=True)
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
