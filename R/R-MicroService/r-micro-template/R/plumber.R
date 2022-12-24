################################################################################
## Pumber.R : users don't touch it
################################################################################

# set API title and description to show up in http://localhost:8000/__swagger__/
#* @apiTitle Run predictions for R ML model
#* @apiDescription This API takes as input data in JSON format to predict.

# load user's main function and needed libraries for predict
source("Interface.R")


#* predict using the request body
#* @serializer unboxedJSON
#* @post /score
function(req){

  input_data <- as.data.frame(lapply(jsonlite::fromJSON(req$postBody), unlist))
  res <- main(input_data)
  resJson <- jsonlite::toJSON(list(data=input_data, result=res), pretty = TRUE)
  return(resJson)
}


#* predict using the post data entry
#* @serializer contentType list(type="application/json")
#* @param data - input data object
#* @post /score-data
function(req, data){

  input_data <- as.data.frame(lapply(jsonlite::fromJSON(data), unlist))
  res <- main(input_data)
  resJson <- jsonlite::toJSON(list(data=input_data, result=res), pretty = TRUE)
  return(resJson)
}


#* Get Microservice health
#* @serializer contentType list(type="application/json")
#* @get /health
function(){
  # s <- '{"status": "UP", "details": {}}'
  s <- '{"status":"UP","details":{"diskSpace":{"status":"UP","details":{}},"refreshScope":{"status":"UP"}}}'
  # jsonlite::toJSON(s)
  s
}