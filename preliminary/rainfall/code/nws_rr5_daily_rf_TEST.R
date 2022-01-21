#this  code grabs 24 hr rainfall total from NWS site from the midnight data upload

rm(list = ls())#remove all objects in R

#set MAIN DIR
mainDir <- "/home/hawaii_climate_products_container/preliminary"

#packages and settings
require(plyr)
options(warn=-1)#suppress warnings for session
print(paste("nws rr5 daily rf agg:",Sys.time()))#for cron log

#dirs
parse_hrly_wd<-paste0(mainDir,"/data_aqs/data_outputs/nws_rr5/parse")
agg_daily_wd<-paste0(mainDir,"/rainfall/working_data/nws_rr5")

#read parse file
setwd(parse_hrly_wd)
nws_filename<-paste0(format((Sys.Date()-1),"%Y%m%d"),"_NWSrr5_parse.csv") #dynamic file name that includes date
nws_hrly_data_final<-read.csv(nws_filename)

#agg to daily and removal partial daily obs
nws_hrly_data_final$date<-as.Date(nws_hrly_data_final$date)#make date a date
nws_hrly_data_final_cc<-nws_hrly_data_final[complete.cases(nws_hrly_data_final),]#remove NA obs
nws_24hr_agg<-ddply(nws_hrly_data_final_cc, .(nwsli,NWS_name, date), summarize, prec_mm_24hr = sum(prec_mm_1hr), hour_count=length(nwsli)) #sum by station & day and count hourly obs per station & day
nws_24hr_agg_final<-nws_24hr_agg[nws_24hr_agg$hour_count >= 23,] #subset by stations with at least 23 hourly obs (ie: only 1 missing data)

#write/append to file
setwd(agg_daily_wd)#set wd to save/append final day aggs
files<-list.files()
rf_month_daily_filename<-paste0("nws_daily_rf_TEST_",format((Sys.Date()-1),"%Y_%m"),".csv")#dynamic filename that includes month year so when month is done new file is written
if(max(as.numeric(files==rf_month_daily_filename))>0){
	write.table(nws_24hr_agg_final,rf_month_daily_filename, row.names=F, sep = ",", col.names = F, append = T)
	}else{
	write.csv(nws_24hr_agg_final,rf_month_daily_filename,row.names=F)
	}
print("Daily NWS data made and saved!")

# CODE PAU!!!!!
