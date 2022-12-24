# Regression: train a model for median house price as a function of the other variables different models

#load libraries
library(randomForest)
library(neuralnet)

#load data
data(Boston, package="MASS")

#linear model
bos.lm <- glm(medv~., data=Boston)
saveRDS(bos.lm, "bos_lm.rds")

#random forest
bos.rf <- randomForest(medv ~ ., data=Boston, ntree=100)
saveRDS(bos.rf, "bos_rf.rds")

#neural net
# 13:5:3:1: input layer (13 inputs), 1st hidden layers (5 neurons), 2nd hidden layer (3 neurons), output layer (1 output)
bos.nn <- neuralnet(medv ~ ., data=Boston, hidden=c(5,3), linear.output=T)
saveRDS(bos.nn, "bos_nn.rds")