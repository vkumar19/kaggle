library(randomForest)
 
data <- train_factor
 
k = 5 #Folds
 
# sample from 1 to k, nrow times (the number of observations in the data)
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k
 
# prediction and testset data frames that we add to with each iteration over
# the folds
 
prediction <- data.frame()
testsetCopy <- data.frame()
formula <- count ~ season + holiday + workingday + weather + daypart + day 

foo = seq(1, 20, 4)
x   = vector()
num = 30

#for (num in foo){ 
	for (i in 1:k){
	# remove rows with id i from dataframe to create training set
	# select rows with id i to create test set
	trainingset <- subset(data, id %in% list[-i])
	testset <- subset(data, id %in% c(i))
	# run a random forest model
	mymodel <- randomForest(formula, data = trainingset, ntree = num)
	# remove response column 12
	temp <- as.data.frame(predict(mymodel, testset[,-12]))
	# append this iteration's predictions to the end of the prediction data frame
	prediction <- rbind(prediction, temp)
	# append this iteration's test set to the test set copy data frame
	# keep only the count length column
	testsetCopy <- rbind(testsetCopy, as.data.frame(testset[,12]))
	}
	 
	# add predictions and actual counts
	result <- cbind(prediction, testsetCopy[, 1])
	names(result) <- c("Predicted", "Actual")
	result$Difference <- (log(result$Predicted + 1) - log(result$Actual + 1))^2
	RMSLE <- sqrt(mean(result$Difference)) 
	# As an example use RMSLE as Evalution
        #append(x, sqrt(mean(result$Difference))) 
#}
