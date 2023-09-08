#this code grabs all raw HADS data for HI and calcs daily RF total from precip accumulations

rm(list = ls())#remove all objects in R

#set options
options(warn=-1)#supress warnings for session
Sys.setenv(TZ='Pacific/Honolulu') #set TZ to honolulu
print(paste("madis rf daily run:",Sys.time()))#for cron log

#set MAIN DIR
mainDir <- "/home/hawaii_climate_products_container/preliminary"
codeDir<-paste0(mainDir,"/rainfall/code/source")

#define dates
source(paste0(codeDir,"/dataDateFunc.R"))
dataDate<-dataDateMkr("2023-05-25") #function for importing/defining date as input or as yesterday
currentDate<-dataDate #dataDate as currentDate (yesterday)

#load packages
#install.packages("xts")
require(xts)

#functions
getmode <- function(v) { #get mode of values
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}#end getmode function
apply.hourly <- function(x, FUN, roundtime = "round", na.rm = TRUE){
  if(!is.xts(x)){
    stop("x must be an xts object")
  }
  
  if(roundtime != "NA"){
    if(roundtime == "round"){
      time(x) <- round.POSIXt(time(x), "hours")
    } else if(roundtime == "trunc"){
      time(x) <- trunc.POSIXt(time(x), "hours")
    } else {
      stop("roundtime must be either round or trunc")
    }
  }
  
  ap <- endpoints(x,'hours')
  if(na.rm){
    period.apply(x,ap,FUN, na.rm = TRUE)
  } else {
    period.apply(x,ap,FUN)
  }
}#end apply.hrly function

#dirs
parse_wd<-paste0(mainDir,"/data_aqs/data_outputs/madis/parse")
agg_daily_wd<-paste0(mainDir,"/rainfall/working_data/madis")

#read MADIS parsed table from ikewai data portal
ikeUrl<-"https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/workflow_data/preliminary_test" #url
madis_filename<-paste0(format((currentDate),"%Y%m%d"),"_madis_parsed.csv") #dynamic file name that includes date
all_madis<-read.csv(paste0(ikeUrl,"/data_aqs/data_outputs/madis/parse/",madis_filename))
#head(all_madis)

hiMesonet<-c('021HI','022HI','027HI','025HI','008HI','018HI','002HI','020HI','017HI','001HI','016HI','028HI','019HI','011HI','012HI','026HI','015HI','014HI','024HI')
staQ<-all_madis[all_madis$stationId %in% hiMesonet,]
unique(staQ$varname)