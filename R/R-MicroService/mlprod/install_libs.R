## This script set up the packrat environment

# ipak function: install and load multiple R packages
# Check to see if packages are installed
# Auto-intallation and loading of libraries
# - Check missing libraries
# - Install them if they are not installed
# - Load them into the R session
#
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
}

### install and load needed R libraries
pkg <- read.csv("requirements.txt", header=FALSE, sep = ",")
ipak(pkg[,])
  