#################################################################################
## The User's R script TEMPLATE: what you want the API to do
#
## This script is modified by the user to create the function main 
## that receives a data.frames (input_data) an returns an object
################################################################################

# ------------------------------------------------------
# PUT YOURS IMPORTS HERE

#read pipeline
pipeline.list <- readRDS(file='/app/pipeline_list.rds')

#load models
library(caret)
# ------------------------------------------------------


# User's main function
# :input: data.frame
# :output: any R object
main <- function(input_data){	
	
	# ---------------------------------------------------
	# PUT YOUR CODE HERE
  
	#prediction
  	res <- pipeline.list$GetNewPredictions(model=pipeline.list$model.obj,
  	                                       preProcessor=pipeline.list$preprocess.obj, 
  	                                       oneHotEncoder=pipeline.list$dummy.obj,
                                           recursiveFE=pipeline.list$rfe.obj, 
                                           newdata=input_data)

	res <- as.factor(gsub("X", "", res))

	return(res)
	# --------------------------------------------------

	}