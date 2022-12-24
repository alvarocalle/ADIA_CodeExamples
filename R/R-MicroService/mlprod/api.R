# API
library(plumber)
r <- plumb("model_scoring.R") 
r$run(port=8000)