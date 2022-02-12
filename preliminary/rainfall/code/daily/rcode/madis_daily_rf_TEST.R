#this code grabs all raw HADS data for HI and calcs daily RF total from precip accumulations

rm(list = ls())#remove all objects in R

#set MAIN DIR
mainDir <- "/home/hawaii_climate_products_container/preliminary"

#set options
options(warn=-1)#supress warnings for session
print(paste("madis rf daily run:",Sys.time()))#for cron log

Sys.setenv(TZ='Pacific/Honolulu') #set TZ to honolulu

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

#read HADS parsed table
setwd(parse_wd)#sever path for parsed hads files
madis_filename<-paste0(format((Sys.Date()-1),"%Y%m%d"),"_madis_parsed.csv") #dynamic file name that includes date
all_madis<-read.csv(madis_filename)

#subset precip var, convert inch to mm and convert UTC to HST
all_madis_pc<-subset(all_madis,varname=="precipAccum" | varname=="rawPrecip")# subset precip only
#all_madis_pc$value<-if(all_madis_pc$value*25.4 #convert to MM
all_madis_pc<-all_madis_pc[complete.cases(all_madis_pc),] #remove NA rows
all_madis_pc$time<-strptime(all_madis_pc$time, format="%Y-%m-%d %H:%M", tz="UTC")
attr(all_madis_pc$time,"tzone") <- "Pacific/Honolulu" #convert TZ attribute to HST
all_madis_pc$time<-(all_madis_pc$time-36000) #minus 10 hrs for UTC to HST
all_madis_pc$time<-(all_madis_pc$time)-1 #minus 1 second to put midnight obs in last day
tail(all_madis_pc)
head(all_madis_pc)

all_madis_pr<-subset(all_madis_pc,varname=="rawPrecip")# subset raw precip only
all_madis_pa<-subset(all_madis_pc,varname=="precipAccum")#subset accum precip only
pr_only_sta<-unique(all_madis_pr$stationId)[!unique(all_madis_pr$stationId) %in% unique(all_madis_pa$stationId)]
sub_madis_pr<-all_madis_pr[all_madis_pr$stationId  %in% pr_only_sta,]
unq_madis_pc<-rbind(sub_madis_pr,all_madis_pa)

#blank DF to store daily data
madis_daily_rf<-data.frame()

#unique hads stations
stations<-unique(unq_madis_pc$stationId)

#start daily RF loop
print("daily rf loop started...")
for(j in stations){
  sta_data<-subset(unq_madis_pc,stationId==j)
  sta_data_xts<-xts(sta_data$value,order.by=sta_data$time,unique = TRUE) #make xtended timeseries object
  sta_data_xts_sub<- sta_data_xts[!duplicated(index(sta_data_xts)),] #remove duplicate time obs
  if(nrow(sta_data_xts_sub)>=23){ #only stations with at least hourly intervals
    sta_data_xts_sub_lag<-diff(sta_data_xts_sub,lag=1)
    sta_data_xts_sub_lag[sta_data_xts_sub_lag<0]<-NA #NA to neg values when lag 1 dif
    sta_data_hrly_xts<-apply.hourly(sta_data_xts_sub_lag,FUN=sum,roundtime = "trunc")#agg to hourly and truncate hour
    # sta_data_hrly_xts<-apply.hourly(sta_data_xts_sub,FUN=sum,roundtime = "trunc")#agg to hourly and truncate hour
    sta_data_daily_xts<-apply.daily(sta_data_hrly_xts,FUN=sum,na.rm = F)#daily sum of all hrly observations
    obs_ints<-diff(index(sta_data_xts_sub),lag=1) #calculate vector of obs intervals
    obs_int_hr<-getmode(as.numeric(obs_ints, units="hours"))
    obs_int_minutes<-obs_int_hr*60
    obs_per_day<-((1/obs_int_hr)*24)#calculate numbers of obs per day based on obs interval
    sta_per_obs_daily_xts<-as.numeric(apply.daily(sta_data_xts_sub_lag,FUN=length)/obs_per_day)#vec of % percentage of obs per day
    # sta_per_obs_daily_xts<-as.numeric(apply.daily(sta_data_xts_sub,FUN=length)/obs_per_day)#vec of % percentage of obs per day
    sta_daily_df<-data.frame(staID=rep(as.character(j),length(sta_data_daily_xts)),date=as.Date(strptime(index(sta_data_daily_xts),format="%Y-%m-%d %H:%M"),format="%Y-%m-%d"),obs_int_mins=rep(obs_int_minutes,length(sta_data_daily_xts)),data_per=sta_per_obs_daily_xts,rf=sta_data_daily_xts)#make df row
    madis_daily_rf<-rbind(madis_daily_rf,sta_daily_df)
  }
}
print("loop complete!")
row.names(madis_daily_rf)<-NULL #rename rows
# head(head(madis_daily_rf))
# tail(madis_daily_rf)

#subsets: yesterday with 95% data
madis_daily_rf_today<-madis_daily_rf[madis_daily_rf$date==(Sys.Date()-1),]#subset yesterday
row.names(madis_daily_rf_today)<-NULL #rename rows
head(madis_daily_rf_today)
tail(madis_daily_rf_today)

madis_daily_rf_today_final<-madis_daily_rf_today[madis_daily_rf_today$data_per>=0.95,]#subset days with at least 95% data
row.names(madis_daily_rf_today_final)<-NULL #rename rows

#final data check
#print(madis_daily_rf_today_final)
head(madis_daily_rf_today_final)
tail(madis_daily_rf_today_final)

#write or append daily rf data monthly file
	#NEED TO INTGRATE WITH IKE DP
setwd(agg_daily_wd)#server path daily agg file
rf_month_filename<-paste0(format((Sys.Date()-1),"%Y_%m"),"_madis_daily_rf.csv") #dynamic file name that includes month year so when month is done new file is written

if(max(as.numeric(list.files()==rf_month_filename))>0){
	 write.table(madis_daily_rf_today_final,rf_month_filename, row.names=F,sep = ",", col.names = F, append = T)
	 print(paste(rf_month_filename,"written"))
      }else{
	 write.csv(madis_daily_rf_today_final,rf_month_filename, row.names=F)
	 print(paste(rf_month_filename,"appended"))
	}

print("PAU!")






 
