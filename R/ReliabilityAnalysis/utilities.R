#### ************************************ ####
#### User defined functions               ####
####                                      ####
#### Upate: 05/10/2017 csv files grouping ####
#### ************************************ ####

# This function returns string w/o leading or trailing whitespace
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

# Multi-replacement function
mgsub <- function(pattern, replacement, x, ...) {
  if (length(pattern)!=length(replacement)) {
    stop("pattern and replacement do not have the same length.")
  }
  result <- x
  for (i in 1:length(pattern)) {
    result <- gsub(pattern[i], replacement[i], result, ...)
  }
  result
}

# This function returns a dataframe with (cleaned) failure data
#
# Inputs:
# - path [string]: "/LTU_RailwayDatabase/WorkOrders/WO.txt"
# - asset [string]: "Track/Switch/Derailer"
# - columns [string-vector]: set of columns names to keep
#
# Remarks:
# - set feature "From_station" as asset identifier
#
# Dependencies:
# - filter{dplyr}
getData <- function(path, asset="Switch", columns){
  
  # Get data:
  data.df <- read.csv(path, strip.white = TRUE)
  
  # Remove starting and ending blank spaces:
  data.df$Asset_type <- trim(as.character(data.df$Asset_type))
  
  # Subset based on asset:
  asset_wo.df <- filter(data.df, Asset_type == asset)
  
  # Remove assets with From_station != To_station
  # Use "From_station" as asset ID
  tmp <- asset_wo.df[, columns]
  sameStation = sum(tmp$From_station == tmp$To_station)
  diffStation = sum(tmp$From_station != tmp$To_station)
  df <- tmp[ tmp$From_station == tmp$To_station , ]
  
  # Sort data-frame by asset ID (From_station):
  df <- df[order(df$From_station), columns]
  row.names(df) <- seq(1:nrow(df))
  
  # Create new column with IDs:
  IDs <- as.character(df$From_station)
  idCounts.df <- as.data.frame(table(IDs), stringsAsFactors=FALSE)
  freq <- as.numeric(idCounts.df[, 2])
  newID <- c()
  count <- 1
  for (num in freq){
    newID <- c(newID,rep(count, num))
    count <- count + 1
  }
  
  # rename IDs
  df$Asset_id <- newID
  
  # Output data-frame
  columns <- append(columns, "Asset_id")
  
  return(df[, columns])
}

# This function plots the Cumulative number of failures
#
# Input:
# - unit.df: data.frame with failure times
#
# Output:
# - p: plot of the cumulative density
# - ct: failure times
# - cf: cumulative number of failures
#
# Dependencies:
# - ggplot {ggplot2}
cumNumFailures <- function(unit.df) {
  
  p <- ggplot(unit.df, aes(x=mgt, group=id)) +
    stat_ecdf(aes(x=mgt, color=factor(id))) +
    labs(x="Cumulative TTF [MGT]", y="Cumulative Dist. of Failures",
         title="", color="ID") +
    theme_bw() +
    theme(legend.position = "right",
          plot.title = element_text(hjust = 0.5),
          panel.grid.minor.x=element_blank(),
          panel.grid.major.x=element_blank())
  
  return(p)
}

# Weibull dist. (expected value)
WeibullMTTF <- function(shape, scale){
  a <- shape
  b <- scale
  MTTF <- b * gamma(1/a + 1)
  return(MTTF)
}

# Weibull dist. (standard deviation)
WeibullSTD <- function(shape, scale){
  a <- shape
  b <- scale
  s <- sqrt( b**2 * ( gamma(2/a + 1) - gamma(1/a + 1)**2) )
  return(sqrt(s))
}

# hazard function z(t) = f(t)/R(t)
Weibullz <- function(t, shape, scale){
  a <- shape
  b <- scale
  z <- dweibull(t, shape, scale)/(1 - pweibull(t, shape, scale))
  # z <- a/b*(t/b)**(a-1)
  return(z)
}

# Cumulative hazard function Z(t): Z'(t) = z(t)
WeibullH <- function(t, shape, scale){
  a <- shape
  b <- scale
  #(t/b)^a
  H <- -pweibull(t, a, b, lower = FALSE, log = TRUE)
  return(H)
}

# Cumulative distribution function F(t)
WeibullF <- function(t, shape, scale){
  a <- shape
  b <- scale
  return(pweibull(t, a, b))
}

# Reliability function R(t) = 1 - F(t)
WeibullR <- function(t, shape, scale){
  # a <- shape
  # b <- scale
  R <- 1 - pweibull(t, shape, scale)
  return(R)
}

# Lognorm dist. (expected value)
LogNormMTTF <- function(mu, sigma){
  MTTF <- exp(mu + sigma**2/2)
  return(MTTF)
}

# Lognorm dist. (standard deviation)
LogNormSTD <- function(mu, sigma){
  s <- sqrt(exp(2*mu + sigma**2) * (exp(sigma**2) - 1))
  return(sqrt(s))
}

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

# ipak function: install and load multiple R packages
# Check to see if packages are installed
# Auto-intallation and loading of libraries
# - Check missing libraries
# - Install them if they are not installed
# - Load them into the R session
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# --------------------------------------------------------------------
# Failure characterisation module
# --------------------------------------------------------------------
# This function calculates the Reliability characteristics
# of the system decomposed by components for a given CM action
# 
# Inputs: 
#
# - maintenance.DT [data.table] = dt with maintenance data
#
#   id [numeric]: asset id
#   time [Date]: failure event time
#   affect_traffic [factor]: if maintenance affected traffic
#   section [integer]: rail section (1,2)
#   part [character]: asset part
#   action [factor]: type of intervention
#   event [numeric]: row of ones needed for survival functions
#
# - action [character string] = maintenance action to be considered
#
# Outputs:
#
# - pFailureEvent [ggplot2]: plot of failure events for action
# - pBoxOutliers [ggplot2]: Tukey box plots of TTF including outliers
# - pBoxNoOutliers [ggplot2]: Tukey box plots of TTF excluding outliers
# - fit.results.Reliab [data.frame]: results for TTF fit with Weibull distro
#   dataframe columns: (part, aW, saW, bW, sbW, mttf, stdv)
#   part= component part
#   aW  = Weibull's shape parameter
#   saW = error in the Weibull's shape parameter
#   bW  = Weibull's scale parameter
#   sbW = error in the Weibull's scale parameter
#   mttf= mean-time-to-failure according to the Weibull
#   stdv= error in the mttf
# --------------------------------------------------------------------
ReliabilityCalculation <- function(maintenance.DT, maintAction){
  
  # filter maintenance action
  DT <- maintenance.DT[action==maintAction]
  
  # action event plot 
  pFailEvent <- ggplot(DT, aes(x=time, y=id, group=section)) +
    geom_point(aes(shape=factor(section)), size = 2) +
    scale_shape_manual(values = c(16,17)) +
    labs(x="Date", y="Asset ID",
         title=paste0("Events: ", maintAction), shape="Section") +
    theme_bw() +
    theme(legend.position = "right", 
          legend.key = element_blank(),
          panel.grid.minor.x=element_blank(),
          panel.grid.major.x=element_blank(),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
          axis.text.y = element_text(angle = 0, hjust = 1, size = 8),
          text = element_text(size=8) ) +
    scale_y_continuous(breaks=seq(min(DT$id), max(DT$id), 1)) +
    scale_colour_gradient(low="red", high="blue")
    
  pFailEvent <- pFailEvent + scale_x_date(
      labels = date_format("%Y-%m-%d"),
      breaks = date_breaks("3 month")) # custom x-axis labels

  # Number of WOs affecting traffic (per action):
  #print(paste0('WOs affecting traffic = ', sum(DT$affect_traffic=='Yes'))) 
  
  # Number of WOs not-affecting traffic (per action):
  #print(paste0('WOs not-affecting traffic = ', sum(DT$affect_traffic=='No')))
  
  ### Failure per component analysis
  # We first define the TTF as the load in MGT from intervention to intervention
  # We select both, events affecting and events non-affecting traffic
  
  #DT <- DT[affect_traffic=='Yes'] # only affecting traffic
  DT <- DT[order(id, part)] # sort by id and part
  DT[, ttf:=c(NA, diff(timestaps)), by=.(id, part)]
  
  # Approach II: consider recurrent and non-recurrent events
  #    - Advantage: more data for the analysis (improve statistics)
  #    - Disadvantage: left-censoring needs to be considered (TBD)
  
  ## DT[is.na(ttf), c("ttf", "event") := list(timestaps, 2)] ##left-censoring effect
  DT[is.na(ttf), ttf := timestaps]
  
  # Now we transform time to MGT:
  # - Ageing takes place due to tonnage accumulation resulting from traffic
  # - Data analysis is based on MGT of traffic flow
  # - The age of the component is calculated by multiplying the annual averaged MGT
  #   of traffic flow with the time difference (in days) from intervention to
  #   intervention (timediff column calculated previously)
  # 
  # From TRV information we have the averaged MGT/year:
  # - MGT/year = 24 (Track section 1)
  # - MGT/year = 17 (Track section 2)
  #
  # We now add a column with Times To Failure (TTF) in MGT
  mgtFactor1 <- 24/365
  mgtFactor2 <- 17/365
  DT[, mgt:=.SD*mgtFactor1, .SDcols = "ttf"][1, mgt:=.SD*mgtFactor2, .SDcols = "ttf"]

  ### MTTF (Approach 2)
  # Considers that ALL components working under **identical** environmental conditions.
  # In order to obtain statistically significant results we select components with at leat five failures.
  
  # minimum number of failures per component:
  nmin = 4
  
  # number of events per component:
  DT <- setDT(DT)[ , if (.N>=nmin) .SD, by=part]
  DT[ , .N, by=part]
  
  # list of statistically significant components
  plist <- unique(DT$part)
  
  ### Boxplots and descriptive statistics. Outliers.
  
  # Outliers (visualization)
  melted.DT <- melt.data.table(DT, id.vars = "part", measure.vars = "mgt")

  pBoxOutliers <- ggplot(data = melted.DT, aes(x=part, y=value)) +
    geom_boxplot(aes(fill=part)) +
    labs(x="Asset part", y="TTF [MGT]",
         title=paste0("TTF for action=", maintAction), fill="Asset Part") +
    theme_bw() +
    theme(legend.position = "right", 
          legend.key = element_blank(),
          panel.grid.minor.x=element_blank(),
          panel.grid.major.x=element_blank(),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(hjust = 1, size = 8),
          axis.text.y = element_text(hjust = 1, size = 8),
          text = element_text(size=8) )

  outliers <- DT[, boxplot.stats(mgt)$out, by=part]
  names(outliers) <- c("part", "mgt")
  
  # Remove outliers
  DT <- DT[!(mgt %in% outliers$mgt)]
  melted.DT <- melt.data.table(DT, id.vars = "part", measure.vars = "mgt")
  
  pBoxNoOutliers <- ggplot(data = melted.DT, aes(x=part, y=value)) +
    geom_boxplot(aes(fill=part)) +
    labs(x="", y="TTF [MGT]",
         title=paste0("TTF for action=", maintAction), fill="Asset Part") +
    theme_bw() +
    theme(legend.position = "right", 
          legend.key = element_blank(),
          panel.grid.minor.x=element_blank(),
          panel.grid.major.x=element_blank(),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(hjust = 0.5, size = 8),
          axis.text.y = element_text(hjust = 0.5, size = 8),
          text = element_text(size=8) )

  # Once we have removed the outliers we calculate the MTTF per type of component. 
  
  # ---> MTTF <---
  results.w <- data.frame(matrix(NA, nrow = length(plist), ncol = 7))
  colnames(results.w) <- c("part", "aW", "saW", "bW", "sbW", "mttf", "stdv")
  
  counter <- 0
  for (p in plist){
    
    counter <- counter + 1
    
    times <- DT[part==p, mgt] # MGT
    #times <- DT[part==p, ttf]  # days
    event <- DT[part==p, event]
    fit.weibull <- survreg(Surv(times, event) ~ 1, dist="weibull")
    
    # model parameters and errors:
    # survreg's scale  =    1/(rweibull shape)
    # survreg's intercept = log(rweibull scale)
    aW <- 1/as.numeric(fit.weibull$scale) # shape
    saW <- sqrt(fit.weibull$var[1,1])/as.numeric(fit.weibull$scale)**2 # shape sd
    bW <- exp(as.numeric(coef(fit.weibull))) # scale
    sbW <- exp(as.numeric(coef(fit.weibull))) * sqrt(fit.weibull$var[2,2]) # scale sd
    
    mttf <- WeibullMTTF(aW, bW)
    stdv <- WeibullSTD(aW, bW)
    
    results.w[counter, ] <- list(p, aW, saW, bW, sbW, mttf, stdv)
  }
  
  return(list(pFailEvent, pBoxOutliers, pBoxNoOutliers, results.w))
}

# --------------------------------------------------------------------
# Maintenance characterisation module
# --------------------------------------------------------------------
# This function calculates the Maintainability characteristics
# of the system decomposed by components for a given CM action
# 
# Inputs: 
#
# - maintenance.DT [data.table] = dt with maintenance data
#
#   id [numeric]: asset id
#   time [float]: down times (NOTICE THE DIFFERENCE W.R.T. RELIABILITY)
#   affect_traffic [factor]: if maintenance affected traffic
#   section [integer]: rail section (1,2)
#   part [character]: asset part
#   action [factor]: type of intervention
#   event [numeric]: row of ones needed for survival functions
#
# - action [character string] = maintenance action to be considered
#
# Outputs:
#
# - pBoxOutliers [ggplot2]: Tukey box plots of TTR including outliers
# - pBoxNoOutliers [ggplot2]: Tukey box plots of TTR excluding outliers
# - fit.results.Maint [data.frame]: results for TTR fit with Log-normal distro
#   dataframe columns: (part, mu, emu, sigma, esigma, mttr, stdv)
#   part   = component part
#   mu     = log-normal mean parameter
#   emu    = error in the log-normal mean parameter
#   sigma  = log-normal standard deviation parameter
#   esigma = error in log-normal standard deviation parameter
#   mttr = mean-time-to-repair according to the lognormal distribution
#   stdv = error in the mttr
# --------------------------------------------------------------------
MaintainabilityCalculation <- function(maintenance.DT, maintAction){
  
  ### filter maintenance action
  DT <- maintenance.DT[action==maintAction]
  
  ## Time to maintain per component analysis: MTTR (Approach 2)
  
  # count number of events per component
  DT[ , .N, by=part]
  
  # minimum number of events per component:
  nmin = 4
  
  # number of events (more than nmin) per component:
  DT <- setDT(DT)[ , if (.N>=nmin) .SD, by=part]
  DT[ , .N, by=part]
  
  # list of statistically significant components
  plist <- unique(DT$part)
  
  ### Boxplots and descriptive statistics. Outliers
  
  # Outliers
  melted.DT <- melt.data.table(DT, id.vars = "part", measure.vars = "time")
  
  pBoxOutliers <- ggplot(data = melted.DT, aes(x=part, y=value)) +
    geom_boxplot(aes(fill=part)) +
    labs(x="Asset part", y="TTR [Hours]",
         title=paste0("TTR for action=", maintAction), fill="Asset Part") +
    theme_bw() +
    theme(legend.position = "right", 
          legend.key = element_blank(),
          panel.grid.minor.x=element_blank(),
          panel.grid.major.x=element_blank(),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(hjust = 1, size = 8),
          axis.text.y = element_text(hjust = 1, size = 8),
          text = element_text(size=8) )
  
  outliers <- DT[, boxplot.stats(time)$out, by=part]
  names(outliers) <- c("part", "time")
  
  # Remove outliers
  DT <- DT[!(time %in% outliers$time)]
  melted.DT <- melt.data.table(DT, id.vars = "part", measure.vars = "time")
  
  pBoxNoOutliers <- ggplot(data = melted.DT, aes(x=part, y=value)) +
    geom_boxplot(aes(fill=part)) +
    labs(x="Asset part", y="TTR [Hours]",
         title=paste0("TTR for action= ", maintAction), fill="Asset Part") +
    theme_bw() +
    theme(legend.position = "right", 
          legend.key = element_blank(),
          panel.grid.minor.x=element_blank(),
          panel.grid.major.x=element_blank(),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(hjust = 1, size = 8),
          axis.text.y = element_text(hjust = 1, size = 8),
          text = element_text(size=8) )
  
  # After removing outliers we further remove *rare* events with TTR > 16
  
  # proportion of TTR > 8hours:
  DT[ , .(prop=round(sum(time > 8)/.N*100)), by=.(part, affect_traffic)]
  
  # LTU discussion: select TTR < 16 hours per component
  DT <- DT[time<16]
  
  # number of events per asset part
  DT[ , .N, by=part]
  
  # outliers
  outliers <- DT[, boxplot.stats(time)$out, by=part]
  names(outliers) <- c("part", "time")
  
  # remove outliers again (in case there are new ones)
  DT <- DT[!(time %in% outliers$time)]
  melted.DT <- melt.data.table(DT, id.vars = "part", measure.vars = "time")
  
  pBoxNoOutliers <- ggplot(data = melted.DT, aes(x=part, y=value)) +
    geom_boxplot(aes(fill=part)) +
    labs(x="Asset part", y="TTR [Hours]",
         title=paste0("TTR for action= ", maintAction), fill="Asset Part") +
    theme_bw() +
    theme(legend.position = "right", 
          legend.key = element_blank(),
          panel.grid.minor.x=element_blank(),
          panel.grid.major.x=element_blank(),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(hjust = 1, size = 8),
          axis.text.y = element_text(hjust = 1, size = 8),
          text = element_text(size=8) )
  
  # Once we have removed the outliers we calculate the MTTR per type of component.
  # We try two different fits: **Weibull** and **Log-normal**.
  
  # MTTR [hours]
  results.w <- data.frame(matrix(NA, nrow = length(plist), ncol = 7))
  results.ln <- data.frame(matrix(NA, nrow = length(plist), ncol = 7))
  
  colnames(results.w) <- c("part", "aW", "saW", "bW", "sbW", "mttr", "stdv")
  colnames(results.ln) <- c("part", "mu", "emu", "sigma", "esigma", "mttr", "stdv")
  
  counter <- 0
  for (p in plist){
    
    counter <- counter + 1
    
    times <- DT[part==p & time>0, time]
    event <- DT[part==p & time>0, event]
    
    fit.weibull <- survreg(Surv(times, event) ~ 1, dist="weibull")
    fit.lognormal <- survreg(Surv(times, event) ~ 1, dist="lognormal")
    
    # -- Weibull model parameters and errors
    # survreg's scale  =    1/(rweibull shape)
    # survreg's intercept = log(rweibull scale)
    aW <- 1/as.numeric(fit.weibull$scale) # shape
    saW <- sqrt(fit.weibull$var[1,1])/as.numeric(fit.weibull$scale)**2 # shape sd
    bW <- exp(as.numeric(coef(fit.weibull))) # scale
    sbW <- exp(as.numeric(coef(fit.weibull))) * sqrt(fit.weibull$var[2,2]) # scale sd
    
    mttr.w <- WeibullMTTF(aW, bW)
    stdv.w <- WeibullSTD(aW, bW)
    
    # -- Log-normal model parameters and errors
    mu <- as.numeric(fit.lognormal$coefficients) 
    emu <- as.numeric(fit.lognormal$var[1,1]) 
    sigma <- as.numeric(fit.lognormal$scale)
    esigma <- as.numeric(fit.lognormal$var[2,2])
    
    mttr.ln <- LogNormMTTF(mu, sigma)
    stdv.ln <- LogNormSTD(mu, sigma)
    
    results.w[counter, ] <- list(p, aW, saW, bW, sbW, mttr.w, stdv.w)
    results.ln[counter, ] <- list(p, mu, emu, sigma, esigma, mttr.ln, stdv.ln)
    
  }
  
  # select log-normal distribution for repair times
  return(list(pBoxOutliers, pBoxNoOutliers, results.ln))
}    

# -----------------------------------------------------------------------
# Availability characterisation module
# -----------------------------------------------------------------------
# This function calculates the Availability characteristics of the system
# from previous Reliability & Maintainability results at component level
# 
# Inputs: 
#
# - reliab.df [data.frame] = df with reliability results
#   part : asset part
#   (aW, saW, bW, sbW) : Weibull parameters
#   mttf : mean-time-to-failure
#   stdv : error in the mttf
#
# - maint.df [data.frame] = df with maintainability results
#   part : asset part
#   (mu, emu, sigma, esigma) : log-normal parameters
#   mttr : mean-time-to-repair
#   stdv : error in the mttr
#
# Output:
#
# - avail.dt [data.frame] : data.frame with availability results per component
#   part = component part
#   AI   = inhirent availability 
# -----------------------------------------------------------------------------
AvailabilityCalculation <- function(reliab.df, maint.df){

  ### Availability calculation per component
  # we can only calculate inherent availability because of the data
  # A_I = MTTF/(MTTF + MTTR)
  
  # list of statistically significant components
  plist <- unique(reliab.df$part)
  
  # create a data.frame with the results
  avail.df <- data.frame(matrix(NA, nrow = length(plist), ncol = 2))
  names(avail.df) <- c("part", "AI")
  avail.df$part <- reliab.df$part
  
  mgtFactor <- 20/365 # average MGT per year
  
  f <- reliab.df$mttf * (1./mgtFactor) * 24. # MTTF in hours
  r <- maint.df$mttr
  
  avail.df$AI <- f/(f+r)*100 # availability in %
  
  return(avail.df)
}

# ----------------------------------------------------------------------------------
# Probabilities of failure are calculated from the hazard function z
# ----------------------------------------------------------------------------------
# If we know that the item is alive at t, the probability that it fails
# in the time interval (t,t+h) is given by the conditional probability
# P( t < T < t+h | T > t ) = z(t)*h, where z(t) is the hazard rate.
# The hazard rate can be calculated with the function Weibullz(t, shape, scale)
# once the fit to survival data has been done.
#
# If we don't know exactly the aging of the item, then we estimate the probability
# of failure from the cumulative distribution function as P(a < T < b) = F(b) - F(a)
# i.e., F(t) = P(T<t) is the probability of failure in a time t.
# 
# Inputs: 
#
# - reliab.df [data.frame] = df with reliability results
#   part : asset part
#   (aW, saW, bW, sbW) : Weibull parameters
#   mttf : mean-time-to-failure
#   stdv : error in the mttf
#
# - falure_time [numeric] : time to failure (in weeks)
#
# Output:
#
# - failprob.dt [data.frame] : df with fail prob in a failure_time by component
# ----------------------------------------------------------------------------------
FailureProbabilityCalculation <- function(reliab.df, failure_time){
  
  # statistically significant components
  plist <- unique(reliab.df$part)

  # create data.frame with results
  results.FailProb <- data.frame(matrix(NA, nrow = length(plist), ncol = 2))
  names(results.FailProb) <- c("part", "fail_prob")
  
  # average MGT per year 
  # (MUST BE THE SAME THAN THE ONE USED IN RELIABILITY CALCULATION)
  mgtFactor <- 20/365

  # Time of failure (from weeks to MGT)
  failure_time <- failure_time * 7 * mgtFactor

  counter <- 0
  for (p in plist){
    
    counter <- counter + 1
    
    aW <- reliab.df[reliab.df$part==p, "aW"]
    bW <- reliab.df[reliab.df$part==p, "bW"]
     
    prob <- WeibullF(failure_time, aW, bW)

    results.FailProb[counter,] <- list(p, prob)
    
  }
  
  return(results.FailProb)
}

# ----------------------------------------------------------------------
# Generate output csv tables for RAMS
#
# Inputs:
#
# - asset[string]: sc (switch and crossing) /t (track)
# - listOfActions[list]: maintenance actions
# - listOfReliabilityResults[list]: results from Reliability calculation
# - listOfMaintainabilityResults[list]: results from Maintainability calculation
# - listOfAvailabilityResults[list]: results from Availability calculation
# - listOfFailureProbabilities[list]: results from Maintainability calculation
#
# Remarks:
# - Tables according to Data Farm in/out tables
# - Tables at component level (sc: S&C/ t: track)
# - MUT = MTTF (mean time to failure)
# - MDT = MTTR (mean time to repair)
# - Theoretically MTTM includes time of corrective + preventive action,
#   therefore for us it's MDT because only corrective have been included
# - Time to failure is in MGT
# - Time to repair is in hours
# - Availability is in %
# - NA stands for Not Applicable/Not calculated
# ----------------------------------------------------------------------
writeCSVFilesRAMS <- function(asset,
                              listOfActions, 
                              listOfReliabilityResults,
                              listOfMaintainabilityResults,
                              listOfAvailabilityResults,
                              listOfFailureProbabilities){
    
  # TK_RAMS
  tk_rams.df <- data.frame(matrix(NA, nrow = 0, ncol = 17))

  # TK_RAMS_PARAMS
  tk_rams_params.df <- data.frame(matrix(NA, nrow = 0, ncol = 6))

  # TK_RAMS_MTTR = TK_RAMS_DOWNTIME
  # We cannot discrimitate MTTR and MDT given current data
  tk_rams_mttr.df <- data.frame(matrix(NA, nrow = 0, ncol = 5))

  # TK_RAMS_FAILURE_PROBABILITY
  tk_rams_probability.df <- data.frame(matrix(NA, nrow = 0, ncol = 3))

  # loop in the list of actions
  for (i in 1:length(listOfActions)){
    
    maintenance_action <- listOfActions[i]
    reliab.df <- listOfReliabilityResults[[i]][[4]]
    maint.df <- listOfMaintainabilityResults[[i]][[3]]
    avail.df <- listOfAvailabilityResults[[i]]
    prob.df <- listOfFailureProbabilities[[i]]
    plist <- unique(reliab.df$part)
    
    df1 <- data.frame(matrix(NA, ncol = 17, nrow = length(plist))) # TK_RAMS
    df2 <- data.frame(matrix(NA, ncol = 6, nrow = length(plist))) # TK_RAMS_PARAMS (MTTF)
    df3 <- data.frame(matrix(NA, ncol = 6, nrow = length(plist))) # TK_RAMS_PARAMS (MTTR)
    df4 <- data.frame(matrix(NA, ncol = 5, nrow = length(plist))) # TK_RAMS_MTTR = TK_RAMS_DOWNTIME
    df5 <- data.frame(matrix(NA, ncol = 3, nrow = length(plist))) # TK_RAMS_FAILURE_PROBABILITY
    
    for (j in 1:length(plist)) {
      
      p <- plist[[j]]
      
      df1[j,] <- c(paste0(asset,"_",p),
                   maintenance_action,
                   "Weibull",
                   "Weibull",
                   "",
                   reliab.df[reliab.df$part==p, "mttf"],
                   reliab.df[reliab.df$part==p, "mttf"],
                   maint.df[maint.df$part==p, "mttr"],
                   avail.df[avail.df$part==p, "AI"],
                   "",
                   "",
                   "Log-normal",
                   "Log-normal",
                   "Log-normal",
                   maint.df[maint.df$part==p, "mttr"],
                   "",
                   "")
    
      df2[j,] <- c(paste0(asset,"_",p),
                   maintenance_action,
                   "Reliability",
                   "MTTF",
                   reliab.df[reliab.df$part==p, "mttf"],
                   reliab.df[reliab.df==p, "stdv"])

      df3[j,] <- c(paste0(asset,"_",p),
                   maintenance_action,
                   "Maintainability",
                   "MTTR",
                   maint.df[maint.df$part==p, "mttr"],
                   maint.df[maint.df$part==p, "stdv"])
      
      df4[j,] <- c(paste0(asset,"_",p),
                   maintenance_action,
                   "Log-normal",
                   maint.df[maint.df$part==p, "mu"],
                   maint.df[maint.df$part==p, "sigma"])
    
      
      df5[j,] <- c(paste0(asset,"_",p),
                   maintenance_action,
                   prob.df[prob.df$part==p, "fail_prob"])
    }
    
    tk_rams.df <- rbind(tk_rams.df, df1)
    tk_rams_params.df <- rbind(tk_rams_params.df, df2, df3)
    tk_rams_mttr.df <- rbind(tk_rams_mttr.df, df4)
    tk_rams_probability.df <- rbind(tk_rams_probability.df, df5)
  }  
    
  # ----------------
  # write CSV files
  # ----------------

  file_name <- paste0("tk_rams.csv")
  colnames(tk_rams.df) <- c("RAMS_ID","FAILURE_MODE",
                            "Reliab", "Fail_rate",
                            "MTBF", "MTTF", "MUT", "MDT",
                            "Inher_avail", "Achiv_avail", "Op_avail",
                            "Maint", "Rep_rate",
                            "MTTM", "MTTR", "HR", "MTBSF")
  write.csv(tk_rams.df, file = paste0("OutputTables/", file_name), row.names = FALSE)
    
  file_name <- paste0("tk_rams_params.csv")
  colnames(tk_rams_params.df) <- c("ID", "FAILURE_MODE",
                                   "TYPE", "NAME", "VALUE", "CI")
  write.csv(tk_rams_params.df, file = paste0("OutputTables/", file_name), row.names = FALSE)
    
  file_name <- paste0("tk_rams_mttr.csv")
  colnames(tk_rams_mttr.df) <- c("ID", "FAILURE_MODE",
                                 "DISTRIBUTION_FAMILY", 
                                 "MU_PARAMETER", 
                                 "SIGMA_PARAMETER")
  write.csv(tk_rams_mttr.df, file = paste0("OutputTables/", file_name), row.names = FALSE)
    
  file_name <- paste0("tk_rams_probability.csv")
  colnames(tk_rams_probability.df) <- c("ID", "FAILURE_MODE", "probability")
  write.csv(tk_rams_probability.df, file = paste0("OutputTables/", file_name), row.names = FALSE)
}

# -------------------------------------------------------------------------
# LCC calculation
# -------------------------------------------------------------------------
# Maintenance cost model
#
# Important remarks:
#
# * The LCC model is built by subdividing the system into the subsystems
#
# * The most frequent maintenance actions are: 
#    - corrective or preventive: adjustment, *replacement* and repair
#    - preventive: tamping, grinding and inspection
#
# * Here we focus on modelling cost for S&C associated to the
#   following maintainance actions on components for which RAMS
#   parameters have been calculated:
#    - Adjustment
#    - Cleaning
#    - Cleanup
#    - Inspection
#    - Lubrication
#    - Provisionally repaired
#    - Repair
#    - Replacement
#    - Restoration
#    - Snow removal
#    - Speed reduction
#    - Taken out of service
#
# The following cost model have been used:
#
# LCC = n_a sum_{i=1}^{K} sum_{j=1}^{N-1} 1/(1+r)^j 
#     * M/MTTF_i {C_{P_i} + MTTR_i (n_{L_i} C_L + C_{E_i})}
#
# where:
#
# * n_a: number of assets
# * K: number of components under analysis
# * N: life period of the asset in years
# * M: Gross Tonnage per year (in MGT)
# * MTTF_i: Mean-Time-To-Failure for component i (in MGT)
# * C_{P_i}: cost of component i (in Euro)
# * MTTR_i: Mean-Time-To-Restore of component i (in hours)
# * n_{L_i}: number of workers needed to carry out the action in component i
# * C_L: labour cost (in Euro/hour)
# * C_{E_i}: cost of equipment nedeed to carry out the action in component i
#
# The quantity M/MTTF represents the frequency of failure. 
# In the following simulation we fix some of these quantities.

# --- Define Replacement Cost Formula ---
#
# Inputs:
#
# - M [float/integer]: average gross tonnage per year (in MGT)
# - N [integer]: life of the asset (number of years)
# - r [float]: discount rate
# - CP [float vector]: set of costs per component (in Euro/hour)
# - CL [float]: labour cost (in Euro/hour)
# - nL [integer vector]: number of worker per type of component
# - CE [float vector]: cost of equipment per type of component
# - MTTF [float vector]: mean time to failure per type of component
# - MTTR [float vector]: mean time to restore per type of component
#
# Output:
# - Total cost in period (in Euro)
# - Individual cost in period (in Euro)
# -------------------------------------------------------------------------
LCC <- function(M, N, r, CP, CL, nL, CE, MTTF, MTTR){
  
  # store lcc per period
  lccPerPeriod <- list()
  
  cost <- 0
  for (j in 1:N){ # annuity sum 
    cost <- cost + M/MTTF*(CP + MTTR*(nL*CL + CE))/(1+r)**j
    l <- list(j, cost, sum(cost))
    lccPerPeriod[[j]] <- list("period"=l[[1]], "componentCost"=l[[2]], "totalCost"=l[[3]])
  }

  return(lccPerPeriod)
}

# ----------------------------------------------------------------------
# Generate output csv tables for LCC
#
# Inputs:
#
# - asset[string]: sc (switch and crossing) /t (track)
# - listOfActions[list]: maintenance actions
# - listOfMCResults[list]: list with the montecarlo results
#
# Remarks:
# - Tables according to Data Farm in/out tables
# ----------------------------------------------------------------------
writeCSVFilesLCC <- function(asset, listOfActions, listOfMCResults){

  # TK_RAMS_COSTS
  tk_rams_costs.df <- data.frame(matrix(NA, nrow = 0, ncol = 6))

  # loop in the list of actions
  for (i in 1:length(listOfActions)){
    
    maintenance_action <- listOfActions[i]
    results.lcc <- listOfMCResults[[i]]
    plist <- names(results.lcc)

    df <- data.frame(matrix(NA, ncol = 6, nrow = length(plist))) # TK_RAMS_COSTS
    
    for (j in 1:length(plist)) {
      
      p <- plist[[j]]
      
      df[j,] <- c(paste0(asset, "_", p),
                  maintenance_action,
                  results.lcc[["mean", p]],
                  results.lcc[["mean", p]] + results.lcc[["sd", p]],
                  results.lcc[["mean", p]] - results.lcc[["sd", p]],
                  results.lcc[["sd", p]])
    }
    
    tk_rams_costs.df <- rbind(tk_rams_costs.df, df)
  }

  file_name <- paste0("tk_rams_costs.csv")
  colnames(tk_rams_costs.df) <- c("ID", "ACTION_ID", "LCC", "UPPER_BOUND", "LOWER_BOUND", "CI")
  write.csv(tk_rams_costs.df, file = paste0("OutputTables/", file_name), row.names = FALSE)
}

# ---------------------------------------------------------------------------
# Function that save plots in local folder
#
# Inputs:
#
# - listOfActions[list]: maintenance actions
# - listOfReliabilityResults[list]: list with the reliability results
# - listOfMaintainabilityResults[list]: list with the maintainability results
# ---------------------------------------------------------------------------
savePlotsRAMS <- function(listOfActions,
                          listOfReliabilityResults,
                          listOfMaintainabilityResults){

  # loop in the list of actions
  for (i in 1:length(listOfActions)){
    
    maintenance_action <- listOfActions[i]
    
    # reliability plots
    for (j in 1:3){
      p <- listOfReliabilityResults[[i]][[j]]
      ggsave(filename = paste0(asset,".ReliabPlot.p",j,".png"), p, device = "png",
             width = 30, height = 20, units = "cm")
    }
      
    # maintainability plots
    for (j in 1:2){
      p <- listOfReliabilityResults[[i]][[j]]
      ggsave(filename = paste0(asset,".MaintPlot.p",j,".png"), p, device = "png",
             width = 30, height = 20, units = "cm")
    }
  }
  #dev.off()
}

finish <- function(msg="\nGood bye my friend ;-p \n"){
  writeLines(msg)
}



