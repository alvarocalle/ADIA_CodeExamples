# **************************************************************************************
# RAMS-LCC Analysis: Rail demonstrator S&C
#
# Description: RAMS & LCC Analysis of S&Cs components
# This script performs RAMS and LCC calculation for S&Cs in INFRALERT's rail demo case
#
# - RAMS are calculated from corrective maintenance actions
# - LCC model is built according to the actions carried out on the assets
#
# Created: 18/04/2017
#
# Updates:
# - 19/09/2017: inclusion of all maintenance actions for S&C
# - 05/10/2017: regrouping of RAMS csv files
#
# Author: Alvaro Calle Cordon
# **************************************************************************************

### First step: set the current directory as the working directory

getwd() # this is the current directory (WHERE YOU ARE)
setwd("<PATH>")

### DB location
DBPATH = "../../LTU_RailwayDatabase/"

### User defined functions
source("utilities.R")

### Load R libraries
pkg <- c("xts",
         #"xlsx",
         #"plyr",
         "dplyr",
         "scales",
         "ggplot2",
         "survival",
         "data.table")
#ipak(pkg)
sapply(pkg, library, character.only = TRUE)

### Data extraction from database

# Now we carry out a series of steps:
# - Import data from csv files
# - Filter the data : asset=Switch
# - Select columns:
#     + Reported_date
#     + Traffic_affecting
#     + Start_date
#     + Finished_date
#     + Track_section
#     + From_station
#     + To_station
#     + Asset_type
#     + Asset_part
#     + Action
# - Rename the asset parts
#     + Control device = Control
#     + Switch (Other) = Other
#     + Conversion Device = Conversion
#     + Switch heating = Heating
#     + Crossing = Crossing
#     + Point blade = Blade
#     + Joint = Joint
#     + snow shield = Snowshield
#     + Ballast = Ballast
#     + guardrail = Guardrail
#     + sleepers = Sleepers
#     + Locking device = Locking
#     + Fortifications = Fortifications
#     + Middle part = Middle
#     + Track = Track

path <- paste0(DBPATH, "WorkOrders/WO.txt")
asset <- "Switch"
colNames <- c("Reported_date",
              "Traffic_affecting",
              "Start_date",
              "Finished_date",
              "Track_section",
              "From_station",
              "To_station",
              "Asset_type",
              "Asset_part",
#              "Actual_fault",
              "Action")
switch.df <- getData(path, asset, colNames)
head(switch.df)

# Rename asset part names
partString <- as.character(switch.df$Asset_part)
unique(partString)

pattern = c("Control device ",
            "Switch \\(Other\\) ",
            "Conversion Device ",
            "Switch heating ",
            "Crossing ",
            "Point blade ",
            "Joint ",
            "snow shield ",
            "Ballast ",
            "guardrail ",
            "sleepers ",
            "Locking device ",
            "Fortifications ",
            "Middle part",
            "Track ")

replacement = c("Control",
                "Other",
                "Conversion",
                "Heating",
                "Crossing",
                "Blade",
                "Joint",
                "Snowshield",
                "Ballast",
                "Guardrail",
                "Sleepers",
                "Locking",
                "Fortifications",
                "Middle",
                "Track")

newString <- mgsub(pattern, replacement, partString)
switch.df$Asset_part <- newString # replace new names

### **********************
### Analysis of failures
### **********************

# From switch.df we create a new data.frame with following failure data:
#   - Asset ID
#   - Reported date
#   - Traffic affecting
#   - Track section
#   - Asset part
#   - Maintenance action
# This data.frame will be used to obtain MTTF/MTBF

# Feature selection
colselec <- c("Asset_id", "Reported_date", "Traffic_affecting",
              "Track_section", "Asset_part", "Action")
df <- switch.df[, colselec ]

#df$event <- sample(c(0,1), nrow(df), replace = T) # randomly select some as censored
df$event <- rep(1, nrow(df)) # select all as failures
names(df) <- c("id", "time", "affect_traffic", "section", "part", "action", "event")
	  
date <- as.POSIXct(df$time)
earliest <- min(date) # "2008-01-03 10:14:51 CET"

# set time origin (first day in 2008)
originT <- as.POSIXct(strptime("2008-01-03 00:00:01", "%Y-%m-%d %H:%M:%S"))

# transform date to numeric as time difference to origin
df$timestaps <- difftime(date, originT, units = "days")

# From all maintenance actions we select to study the following:
#
# - Adjustment 
# - Cleaning 
# - Cleanup
# - Inspection 
# - Lubrication 
# - Provisionally repaired 
# - Repair 
# - Replacement 
# - Restoration 
# - Snow removal 

listOfActions <- c("Adjustment",
                   "Cleaning", 
                   "Cleanup",
                   "Inspection",
                   "Lubrication",
                   "Provisionally repaired",
                   "Repair", 
                   "Replacement", 
                   "Restoration"
                   #, 
                   #"Snow removal"
                   )

# create data.table and filter the relevant actions
maintenance.DT <- as.data.table(df)
maintenance.DT <- maintenance.DT[action %in% listOfActions]
maintenance.DT$time <- as.Date(maintenance.DT$time)

# proportion of interventions:
maintenance.DT[ , round(.N/nrow(maintenance.DT)*100), by = action]

# Event plot by action:
p <- ggplot(maintenance.DT, 
            aes(x=maintenance.DT$time,
                y=maintenance.DT$id,
                group=maintenance.DT$action)) +
  geom_point(aes(color=factor(maintenance.DT$action)), size = 2) +
#  scale_shape_manual(values = c(15:18,0:2,5)) +
  labs(x="Date", y="ID",
       title="Event plot: failure occurrences in Track",
       color="Action") +
  theme_bw() +
  theme(legend.position = "right", 
        legend.key = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 10),
        panel.grid.minor.x=element_blank(),
        panel.grid.major.x=element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 8, hjust = 1) )# +
  #scale_colour_gradient(low="red", high="blue")

p <- p + scale_x_date(
  labels = date_format("%Y-%m-%d"),
  breaks = date_breaks("6 months")) # custom x-axis labels

ggsave(filename = paste0(asset,".eventplot.png"), p, device = "png",
       width = 30, height = 20, units = "cm")

# --------------------------------------------------------------
# Store the results in a list of lists: listOfReliabilityResults
#
# The output of results is as follows: listOfReliabilityResults[[i]][[j]]
# i = type of action
# j = (Event, TukeyBox, CleanTukeyBox, Reliability)
# ---------------------------------------------------------------
listOfReliabilityResults <- list()

## Here we loop on listOfActions:
for (i in 1:length(listOfActions)){
  
  print(paste('action =', listOfActions[[i]]))
  res <- ReliabilityCalculation(maintenance.DT, listOfActions[[i]])

  listOfReliabilityResults[[i]] <- list(res[[1]], res[[2]], res[[3]], res[[4]])
}

### *************************
### Analysis of maintenance
### *************************
  
# From switch.df we create a new data.frame with following failure data:
#   - Asset ID
#   - Start date
#   - Finished date
#   - Traffic affecting
#   - Track section
#   - Asset part
#   - Maintenance action
#
# Important remarks:
#
#   * The database does not provide information about repair times repair
#     and logistic times cannot be disentangled. Therefore maintainability
#     is measured by Mean-Time-To-Restore (MTTR).
#
#   * The TTR are calculated as the time-difference between the WO starts and ends.
#
#   * Unfortunatelly, maintenance crew do not report whether the action is partly
#     executed and finished on a later occasion, which may result in non-representative
#     restoration times in some cases.
#
#   * Only maintenance activities with less-than-16-man-hours are kept.
#     Unusual long times are considered as outliers and filtered out from the database. 
#
#   * The calculation of MTTR is carried out using a log-normal distribution.

# Feature selection
colselec <- c("Asset_id", "Start_date", "Finished_date", 
              "Traffic_affecting", "Track_section", "Asset_part", "Action")
df <- switch.df[, colselec ]

df$time <- as.numeric(difftime(as.POSIXct(df$Finished_date),as.POSIXct(df$Start_date), units = "hours"))
df$event <- rep(1, nrow(df)) # select all as failures
df <- df[c("Asset_id", "time", "Traffic_affecting", 
           "Track_section", "Asset_part", "Action", "event")]
names(df) <- c("id", "time", "affect_traffic", "section", "part", "action", "event")

# create data.table and filter the relevant actions
maintenance.DT <- as.data.table(df)
maintenance.DT <- maintenance.DT[action %in% listOfActions]
maintenance.DT <- maintenance.DT[order(id, part)] # sort by id and part

# --------------------------------------------------------------
# Store the results in a list of lists: listOfMaintainabilityResults
#
# The output of results is as follows: listOfMaintainabilityResults[[i]][[j]]
# i = type of action
# j = (TukeyBox, CleanTukeyBox, Maintainability)
listOfMaintainabilityResults <- list()

## Here we loop on listOfActions:
for (i in 1:length(listOfActions)){
  
  print(paste('action =', listOfActions[[i]]))
  res <- MaintainabilityCalculation(maintenance.DT, listOfActions[[i]])
  
  listOfMaintainabilityResults[[i]] <- list(res[[1]], res[[2]], res[[3]])
}

### ***************************************************
### Analysis of availability & probability of failures
### ***************************************************

# -----------------------------------------------------------------
# Store the results in lists: 
#
# * listOfAvailabilityResults[[i]]
#   i = type of action
#   return data.frame with (part, AI) for action type i
#
# * listOfFailureProbabilities[[i]]
#   i = type of action
#   return data.frame with (part, Prob) for action type i
# 
# -----------------------------------------------------------------
listOfAvailabilityResults <- list()
listOfFailureProbabilities <- list()

time_to_failure <- 1 # weeks

for (i in 1:length(listOfActions)){
  
  reliab.df <- listOfReliabilityResults[[i]][[4]]
  maint.df <- listOfMaintainabilityResults[[i]][[3]]

  # availability
  res1 <- AvailabilityCalculation(reliab.df, maint.df)
  listOfAvailabilityResults[[i]] <- res1
  
  # probability of failure
  res2 <- FailureProbabilityCalculation(reliab.df, time_to_failure)
  listOfFailureProbabilities[[i]] <- res2

}

### ************************************
### Generate output csv tables for RAMS
### ************************************
writeCSVFilesRAMS(asset, 
                  listOfActions,
                  listOfReliabilityResults,
                  listOfMaintainabilityResults,
                  listOfAvailabilityResults,
                  listOfFailureProbabilities)

### ************************************
### Save plots from RAMS
### ************************************
savePlotsRAMS(listOfActions,
              listOfReliabilityResults,
              listOfMaintainabilityResults)

# We now can estimate the costs:

### ************************************
### LCC calculation
### ************************************

# --- Fixed parameters ---

# Average gross tonnage per year:
M = 20 # MGT

# Life of the switch (600 MGT)
N = 24 # years

# Discount rate 
r = 0.04 # 4%

# Average cost per component (CP) in monetary units (mu):
CP_conversion = 10000 # mu/unit
CP_control    = 10000 # mu/unit
CP_heating    = 10000 # mu/unit
CP_crossing   = 10000 # mu/unit
CP_blade      = 10000 # mu/unit

# LCC for replacement:
CP <- 10000
### we may include different cost per component (in this case we don't)
### in which case we have to account for vector dimensions in the LCC formula
### CP <- c(CP_conversion, CP_control, CP_heating, CP_crossing, CP_blade)

# Average labour cost in monetary units (mu)
CL = 1 # mu/hour

# Number of workers
nL = 3

# Equipment cost for actions in monetary units (mu)
CE = 5 # mu/hour (fix for all diff actions)

# store individual LCC in a list
listOfLCCResults <- list()

# loop in the list of actions
for (i in 1:length(listOfActions)){
  
  # RAMS results
  maintenance_action <- listOfActions[i]
  reliab.df <- listOfReliabilityResults[[i]][[4]]
  maint.df <- listOfMaintainabilityResults[[i]][[3]]
  avail.df <- listOfAvailabilityResults[[i]]
  prob.df <- listOfFailureProbabilities[[i]]
  plist <- unique(reliab.df$part)
  
  MTTF <- reliab.df$mttf
  MTTR <- maint.df$mttr

  if (maintenance_action %in%c("Repair", "Replacement", "Restoration")){
    listOfLCCResults[[i]] <- LCC(M, N, r, CP, CL, nL, CE, MTTF, MTTR)
  } else {
    listOfLCCResults[[i]] <- LCC(M, N, r, 0, CL, nL, CE, MTTF, MTTR)
  }
}
  
# Total cost per period
lcc.df <- data.frame(matrix(NA, nrow = N, ncol = 1 + length(listOfActions)))
colnames(lcc.df) <- c("period", listOfActions)

# populate data.frame with Lcc results
for (i in 1:N){
  lcc.df[i,1] <- i
  for (j in 2:(length(listOfActions)+1)){
    lcc.df[i,j] <- sapply(listOfLCCResults, `[`, i)[[j-1]][[3]]
  }
}

melted.df <- melt(lcc.df, id = "period")

p <- ggplot(melted.df, aes(x=period, y=value/1000, fill=variable))+
            #, fill=lcc.df$variable)) +
  geom_bar(stat="identity") +
  labs(title = "Maintenance costs in 24 life-cycle periods",
       x="Period", y="Accumulated costs (1000 m.u.)", fill = "Maintenance Action:") +
  theme_bw() +
  theme(legend.position = "right", 
        legend.key = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 14),
        text = element_text(size=14),
        axis.ticks.x = element_blank(),
        axis.text.x = element_text(angle = 0, hjust = 0.5),
        axis.text.y = element_text(angle = 0, hjust = 0.5, size = 14))

ggsave(filename = paste0(asset,".lccplot.png"), p, device = "png",
       width = 30, height = 20, units = "cm")

### ************************************
### LCC Monte Carlo simulation
### ************************************
#
# MC simulation to reflect in the LCC the variability of the cost elements.
#
# Assumptions in the simulation:
#
#  * Fixed variables: $n_s$, $K$, $N$, $C_{P_i}$, $n_{L_i}$, $C_{E_i}$, $C_L$
#  * $M$ will be assumed to be normally distributed around $20$ MGT
#  * $MTTF$ is Weibull distributed with different parameters for each component
#  * $MTTR$ is log-normally distributed with different parameters for each component

# --- Fixed parameters ---

# Life of the switch (600 MGT)
N = 24 # years

# Discount rate 
r = 0.04 # 4%

# Average cost per component (CP) in monetary units (mu):
CP_conversion = 10000 # mu/unit
CP_control    = 10000 # mu/unit
CP_heating    = 10000 # mu/unit
CP_crossing   = 10000 # mu/unit
CP_blade      = 10000 # mu/unit

# LCC for replacement:
CP <- 10000
### we may include different cost per component (in this case we don't)
### in which case we have to account for vector dimensions in the LCC formula
### CP <- c(CP_conversion, CP_control, CP_heating, CP_crossing, CP_blade)

# Average labour cost in monetary units (mu)
CL = 1 # mu/hour

# Number of workers
nL = 3

# Equipment cost for actions in monetary units (mu)
CE = 5 # mu/hour (fix for all diff actions)

# --- variable parameters --- 

# Average gross tonnage per year:
M0 = 20 # MGT

# Number of MC points
nmc = 100

# store individual MC results in a list
listOfMCResults <- list()

# loop in the list of actions
for (i in 1:length(listOfActions)){
  
  # RAMS results
  maintenance_action <- listOfActions[i]
  reliab.df <- listOfReliabilityResults[[i]][[4]]
  maint.df <- listOfMaintainabilityResults[[i]][[3]]
  plist <- unique(reliab.df$part)
  
  # Weibull parameters of component
  aW <- reliab.df$aW
  bW <- reliab.df$aW
  
  # Log-normal parameters of component
  mu <- maint.df$mu
  sigma <- maint.df$sigma
  
  # LCC for this action  
  lcc.df <- data.frame(matrix(0, nrow = nmc, ncol = length(plist)))
  names(lcc.df) <- plist
  
  for (j in 1:nmc){
    
    # Generate one load
    M = rnorm(1, M0, M0*0.1)
    
    # Generate one MTTF for each component (Weibull)
    MTTF <- mapply(function(x,y) rweibull(1, shape=x, scale=y), x=aW, y=bW)
    
    # Generate one MTTR for each component (Lognormal)
    MTTR <- mapply(function(x,y) rlnorm(1, meanlog = x, sdlog = y), x=mu, y=sigma)
    
    # Accumulated cost in the last period
    if (maintenance_action %in%c("Repair", "Replacement", "Restoration")) {
      ans <- LCC(M, N, r, CP, CL, nL, CE, MTTF, MTTR)[[N]]
    } else {
      ans <- LCC(M, N, r, 0, CL, nL, CE, MTTF, MTTR)[[N]]
    }

    lcc.df[j, ] <- ans$componentCost
  }
  
  # Calculate mean and variance of cost per component
  listOfMCResults[[i]] <- as.data.frame(sapply(lcc.df, function(cl) list(mean=mean(cl,na.rm=TRUE), sd=sd(cl,na.rm=TRUE))))
}

### ************************************
### Generate output csv tables for LCC
### ************************************
writeCSVFilesLCC(asset, listOfActions, listOfMCResults)

finish()
