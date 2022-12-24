# set API title and description to show up in http://localhost:8000/__swagger__/
#* @apiTitle Run predictions for Boston Dataset with different models
#* @apiDescription This API takes as input Boston data to predict the price of a house depending on several variables.
#* For details on the data, see http://math.furman.edu/~dcs/courses/math47/R/library/mlbench/html/BostonHousing.html

#load libraries
library(randomForest)
library(neuralnet)

#load models
bos.lm <- readRDS("bos_lm.rds")
bos.rf <- readRDS("bos_rf.rds")
bos.nn <- readRDS("bos_nn.rds")

# #* Log system time, request method and HTTP user agent of the incoming request
# #* @filter logger
# function(req){
#   cat("System time:", as.character(Sys.time()), "\n",
#       "Request method:", req$REQUEST_METHOD, req$PATH_INFO, "\n",
#       "HTTP user agent:", req$HTTP_USER_AGENT, "@", req$REMOTE_ADDR, "\n")
#   plumber::forward()
# }

#* predict house price in Boston with different models
#* @param model - string model type
#* @param crim - per capita crime rate by town
#* @param zn - proportion of residential land zoned for lots over 25,000 sq.ft
#* @param indus - proportion of non-retail business acres per town
#* @param chas - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#* @param nox - nitric oxides concentration (parts per 10 million)
#* @param rm - average number of rooms per dwelling
#* @param age - proportion of owner-occupied units built prior to 1940
#* @param dis - weighted distances to five Boston employment centres
#* @param rad - index of accessibility to radial highways
#* @param tax - full-value property-tax rate per USD 10,000
#* @param ptratio - pupil-teacher ratio by town
#* @param black - 1000(B - 0.63)^2 where B is the proportion of blacks by town
#* @param lstat - percentage of lower status of the population
#* @param medv - median value of owner-occupied homes in USD 1000's
#* @post /score
function(model, crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat){

  #input data frame
  input_data <- data.frame(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat)
  
  if (model=='Random Forest'){
    predict(bos.rf, input_data)
  }
  else if (model=='Neural Network'){
    predict(bos.nn, input_data)
  }
  else {
    predict(bos.lm, input_data)
  }
}