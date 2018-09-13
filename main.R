#import data by a faster way#
library(data.table)
train = fread("train.csv", showProgress = TRUE)
test = fread("test.csv", showProgress = TRUE)

#transfer train loss data#
ID = 'id'
Loss = 'loss'
y_train = log(train[,Loss, with = FALSE])[[Loss]]

#remove first and last column for train, and remove first for test#
train$id=NULL
train$loss=NULL
test$id=NULL
train_row_size = nrow(train)
train_test = rbind(train, test)
name = names(train)

for (i in name) {
  if (class(train_test[[i]])=="character") {
    levels <- unique(train_test[[i]])
    train_test[[i]] <- as.integer(factor(train_test[[i]], levels=levels))
  }
}

x_train = train_test[1:train_row_size,]
x_test = train_test[(train_row_size+1):nrow(train_test),]

library(Matrix)
library(xgboost)
library(Metrics)
data_train = xgb.DMatrix(as.matrix(x_train), label=y_train)
data_test = xgb.DMatrix(as.matrix(x_test))

#regression parameter#
para_learning = list(
  seed = 0,
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.075,
  objective = 'reg:linear',
  max_depth = 6,
  num_parallel_tree = 1,
  min_child_weight = 1,
  base_score = 7
)

xgboosting_evaluation <- function (yhat, data_train) {
  y = getinfo(data_train, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

#do iteration to find best parameter#
error = xgb.cv(para_learning,
             data_train,
             nrounds=750,
             nfold=4,
             early_stopping_rounds=100,
             print_every_n = 10,
             verbose= 1,
             feval=xgboosting_evaluation,
             maximize=FALSE)

best_nrounds = error$best_iteration
gbdt = xgb.train(para_learning, data_train, best_nrounds)
prediction = fread("sample_submission.csv", colClasses = c("integer", "numeric"))
prediction$loss = exp(predict(gbdt,data_test))
View(x=prediction$loss)
sample=read.csv("sample_submission.csv")
first_column=sample[,1]
MyData=cbind.data.frame(first_column,prediction$loss)
write.csv(MyData,file = "MyData.csv",row.names=FALSE)

