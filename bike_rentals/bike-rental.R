setwd("~/vaibhaw/kaggle/bike_rental")

#read in train/test
train <- read.csv("train.csv")
test <- read.csv("test.csv")

#str(train) #alternative to summary

#factorize training set
train_factor <- train
train_factor$weather <- factor(train$weather)
train_factor$holiday <- factor(train$holiday)
train_factor$workingday <- factor(train$workingday)
train_factor$season <- factor(train$season)

#factorize test set
test_factor <- test
test_factor$weather <- factor(test$weather)
test_factor$holiday <- factor(test$holiday)
test_factor$workingday <- factor(test$workingday)
test_factor$season <- factor(test$season)

#create time column by stripping out timestamp
#substring(x, start, stop)
train_factor$time <- substring(train$datetime,12,20)
test_factor$time <- substring(test$datetime,12,20)
#str(train_factor)

#factorize new timestamp column
train_factor$time <- factor(train_factor$time)
test_factor$time <- factor(test_factor$time)

#create day of week column
train_factor$day <- weekdays(as.Date(train_factor$datetime))
train_factor$day <- as.factor(train_factor$day)
test_factor$day <- weekdays(as.Date(test_factor$datetime))
test_factor$day <- as.factor(test_factor$day)
aggregate(train_factor[,"count"],list(train_factor$day),mean)

#convert time and create $hour as integer to evaluate
train_factor$hour<- as.numeric(substr(train_factor$time,1,2))
test_factor$hour<- as.numeric(substr(test_factor$time,1,2))

#create daypart column, default to 4 to make things easier for ourselves
train_factor$daypart <- "5"
test_factor$daypart <- "5"

#12AM - 6AM = 1
train_factor$daypart[(train_factor$hour <= 6) & (train_factor$hour >= 0)] <- 1
test_factor$daypart[(test_factor$hour <= 6) & (test_factor$hour >= 0)] <- 1
#7AM - 9AM = 2
train_factor$daypart[(train_factor$hour <= 9) & (train_factor$hour > 6)] <- 2
test_factor$daypart[(test_factor$hour <= 9) & (test_factor$hour > 6)] <- 2
#10AM - 4PM = 3
train_factor$daypart[(train_factor$hour <= 16) & (train_factor$hour > 9)] <- 3
test_factor$daypart[(test_factor$hour <= 16) & (test_factor$hour > 9)] <- 3
#5PM - 7PM = 4
train_factor$daypart[(train_factor$hour <= 19) & (train_factor$hour > 16)] <- 4
test_factor$daypart[(test_factor$hour <= 19) & (test_factor$hour > 16)] <- 4

#convert daypart to factor
train_factor$daypart <- as.factor(train_factor$daypart)
test_factor$daypart <- as.factor(test_factor$daypart)

#convert hour back to factor
train_factor$hour <- as.factor(train_factor$hour)
test_factor$hour <- as.factor(test_factor$hour)
str(train_factor)
