#this code grabs all HADS data for HI, saves multiple copies, converts to MM and calcs daily RF total from precip acculations

rm(list = ls())#remove all objects in R

options(warn=-1)#supress warnings for session
print(paste("hads_24hr webscape run:",Sys.time()))#for cron log

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

#set up dirs:server
raw_page_wd<-"/home/mplucas/precip_pipeline_container/final_scripts/workflows/dailyDataGet/HADS/outFiles/raw"
parse_wd<-"/home/mplucas/precip_pipeline_container/final_scripts/workflows/dailyDataGet/HADS/outFiles/parse"
agg_daily_wd<-"/home/mplucas/precip_pipeline_container/final_scripts/workflows/dailyDataGet/HADS/outFiles/agg"
 
 
#get data from HADS url
url<-"https://hads.ncep.noaa.gov/nexhads2/servlet/DecodedData?sinceday=3&hsa=nil&state=HI&nesdis_ids=nil&of=1" #this is the URL for all data in website in the last 24
page<-readLines(url)

#save raw data
setwd(raw_page_wd)#sever path for un-parsed hads files

page_name<-paste0("hads_hi_page_start",format(Sys.Date()-3,"%Y_%m_%d"),"_end",format(Sys.Date()-1,"%Y_%m_%d"),"_1am.txt")
writeLines(page,page_name)
print(paste("web-scrape saved...",page_name))

#make df of all vars and save/append to csv
all_hads<- read.table(textConnection(page),header=F, sep="|") #this reads that HADS url
names(all_hads)<-c("staID","NWS_sid","var","obs_time","value","random","null")
all_hads$null<-NULL
head(all_hads,3)

#write or append data to all data parsed table
setwd(parse_wd)#sever path for parsed hads files

unagg_hads_monthly_filename<-paste0(format((Sys.Date()-1),"%Y_%m"),"_hads_1am_all_data.csv")#dynamic filename that includes month year so when month is done new file is writen

files<-list.files()
if(max(as.numeric(files==unagg_hads_monthly_filename))>0){
	write.table(all_hads,unagg_hads_monthly_filename, row.names=F,sep = ",", col.names = F, append = T)
	}else{
	write.csv(all_hads,unagg_hads_monthly_filename, row.names=F)
	}
print("pased all data table saved...")

#subset vars and cols and convert to mm
all_hads_pc<-subset(all_hads,var=="PC")# subset precip only
all_hads_pc$random<-trimws(as.character(all_hads_pc$random))
all_hads_pc<-subset(all_hads_pc, random == "")[,-c(6)] #remove random samples and random col
all_hads_pc$value<-all_hads_pc$value*25.4 #convert to MM
all_hads_pc<-all_hads_pc[complete.cases(all_hads_pc),] #remove NA rows
all_hads_pc$obs_time<-strptime(all_hads_pc$obs_time, format="%Y-%m-%d %H:%M", tz="UTC")
attr(all_hads_pc$obs_time,"tzone") <- "Pacific/Honolulu" #convert TZ attribute to HST
all_hads_pc$obs_time<-(all_hads_pc$obs_time-36000) #minus 10 hrs for UTC to HST
all_hads_pc$obs_time<-(all_hads_pc$obs_time)-1 #minus 1 second to put midnight obs in last day
tail(all_hads_pc,3)
head(all_hads_pc,3)

#blank DF to store daily data 
hads_daily_rf<-data.frame()

#unique hads stations
stations<-unique(all_hads_pc$staID)

#start daily RF loop
print("daily rf loop started...")
for(j in stations){
  sta_data<-subset(all_hads_pc,staID==j)
  sta_data_xts<-xts(sta_data$value,order.by=sta_data$obs_time,unique = TRUE) #make xtended timeseries object
  sta_data_xts_sub<- sta_data_xts[!duplicated(index(sta_data_xts)),] #remove duplicate time obs
  if(nrow(sta_data_xts_sub)>=23){
    sta_data_xts_sub_lag<-diff(sta_data_xts_sub,lag=1)
    sta_data_xts_sub_lag[sta_data_xts_sub_lag<0]<-NA #NA to neg values when lag 1 dif
    sta_data_hrly_xts<-apply.hourly(sta_data_xts_sub_lag,FUN=sum,roundtime = "round")#agg to hourly and round to nearest hour
    sta_data_daily_xts<-apply.daily(sta_data_hrly_xts,FUN=sum,na.rm = F)#daily sum of all all lag observations
    obs_ints<-diff(index(sta_data_xts_sub),lag=1) #calculate vector of obs intervals
    obs_int_hr<-getmode(as.numeric(obs_ints, units="hours"))
    obs_int_minutes<-obs_int_hr*60
    obs_per_day<-((1/obs_int_hr)*24)#calculate numbers of obs per day based on obs interval
    sta_per_obs_daily_xts<-as.numeric(apply.daily(sta_data_xts_sub_lag,FUN=length)/obs_per_day)#vec of % percentage of obs per day
    sta_daily_df<-data.frame(NWS_sid=rep(as.character(unique(sta_data$NWS_sid)),length(sta_data_daily_xts)),staID=rep(as.character(j),length(sta_data_daily_xts)),date=as.Date(strptime(index(sta_data_daily_xts),format="%Y-%m-%d %H:%M"),format="%Y-%m-%d"),obs_int_mins=obs_int_minutes,data_per=sta_per_obs_daily_xts,rf=sta_data_daily_xts)#make df row
    hads_daily_rf<-rbind(hads_daily_rf,sta_daily_df)
  }
}
print("loop complete!")
row.names(hads_daily_rf)<-NULL #rename rows

head(hads_daily_rf)
tail(hads_daily_rf)

#subsets: yesterday with 95% data
hads_daily_rf_today<-hads_daily_rf[hads_daily_rf$date==(Sys.Date()-1),]#subset yesterday
row.names(hads_daily_rf_today)<-NULL #rename rows
head(hads_daily_rf_today)
tail(hads_daily_rf_today)

hads_daily_rf_today_final<-hads_daily_rf_today[hads_daily_rf_today$data_per>=0.95,]#subset days with at least 95% data
row.names(hads_daily_rf_today_final)<-NULL #rename rows

#final data check
#print(hads_daily_rf_today_final)
head(hads_daily_rf_today_final)
tail(hads_daily_rf_today_final)

#write or append daily rf data monthly file
setwd(agg_daily_wd)#server path daily agg file
rf_month_filename<-paste0("hads_daily_rf_1am","_",format((Sys.Date()-1),"%Y_%m"),".csv")#dynamic filename that includes month year so when month is done new file is written

if(max(as.numeric(list.files()==rf_month_filename))>0){
	 write.table(hads_daily_rf_today_final,rf_month_filename, row.names=F,sep = ",", col.names = F, append = T)
      }else{
	 write.csv(hads_daily_rf_today_final,rf_month_filename, row.names=F)
	}

print("PAU!")






 