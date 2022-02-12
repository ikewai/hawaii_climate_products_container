#this code combines all daily agg rf together and removes duplicates by priority ranking
rm(list = ls())#remove all objects in R

#set MAIN DIR
mainDir <- "/home/hawaii_climate_products_container/preliminary"
#mainDir <- "C:/Users/Matt Lucas/Documents/jetsream_backup/" #dev dir

options(warn=-1)#suppress warnings for session
print(paste("all data daily merge run:",Sys.time()))#for cron log

#functions
rbind.all.columns <- function(x, y) {     #function to smart rbind
    x.diff <- setdiff(colnames(x), colnames(y))
    y.diff <- setdiff(colnames(y), colnames(x))
    x[, c(as.character(y.diff))] <- NA 
    y[, c(as.character(x.diff))] <- NA 
    return(rbind(x, y))}

#input dirs
hads_daily_wd <- paste0(mainDir,"/rainfall/working_data/hads") #hads daily agg data wd
nws_daily_wd <- paste0(mainDir,"/rainfall/working_data/nws_rr5") #nws daily agg wd
scan_daily_wd <- paste0(mainDir,"/rainfall/working_data/scan") #scan daily agg wd
#WORKING... madis daily agg wd
#WORKING... hi mesonet daily agg wd
#WORKING... smart ala wai daily agg wd

#output dirs
missing_sta_wd <- paste0(mainDir,"/rainfall/data_outputs/tables/rf_station_tracking/missing") #unknown stations
count_log_wd <- paste0(mainDir,"/rainfall/data_outputs/tables/rf_station_tracking/count") #station counts per data stream per day
rf_day_source_wd <- paste0(mainDir,"/rainfall/data_outputs/tables/station_data/daily/source/statewide") #datastream source of data
rf_day_data_wd <- paste0(mainDir,"/rainfall/data_outputs/tables/station_data/daily/raw/statewide") #final combine daily rainfall data

#add master metadata with SKN and lat long
meta_url <- "https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv"
geog_meta<-read.csv(meta_url, colClasses=c("NESDIS.id"="character"))

#names(geog_meta)
#str(geog_meta)
geog_meta_sub<-geog_meta[,c("SKN","NESDIS.id","SCAN.id","NWS.id","SMART_NODE_RF.id","LAT","LON")]
geog_meta_sub$no_sourceID<-geog_meta_sub$SKN
#head(geog_meta_sub)
print("master meta added!")

#define date
map_date<-Sys.Date()-1# yesterday as date
file_date<-format(map_date,"%Y_%m")

#add HADS data
setwd(hads_daily_wd)#set data source wd
hads_month_filename<-paste0(file_date,"_hads_daily_rf.csv")#dynamic file name that includes month year so when month is done new file is written
if(max(as.numeric(list.files()==hads_month_filename))>0){  #does HADS month file exist? 
 hads<-read.csv(paste0(hads_month_filename),header=T,colClasses=c("staID"="character"))
 hads<-hads[,c("staID","date","rf")]
 names(hads)<-c("sourceID","date","x") #note 'source_id" IS "NESDIS.id" for hads
 hads$date<-as.Date(hads$date)
 hads_day<-hads[hads$date==map_date,] #date sub
 if(nrow(hads_day)>0){ #if hads_day has rows/data
  hads_day$date<-format(hads_day$date,"%Y.%m.%d")
  hads_day<-hads_day[,c("sourceID","date","x")]
  hads_day$x<-as.numeric(hads_day$x)
  #head(hads_day)
  hads_wide<- reshape(hads_day, idvar = "sourceID", timevar = "date", direction = "wide")
  hads_wide$datastream<-"hads"
  #tail(hads_wide)
  hads_wide_merged_all<-merge(hads_wide,geog_meta_sub[,c("NESDIS.id","SKN")],by.x="sourceID",by.y="NESDIS.id",all.x=T)
  missing_hads<-hads_wide_merged_all[is.na(hads_wide_merged_all$SKN),c("sourceID","datastream")] #missing stations
  hads_wide_merged<-hads_wide_merged_all[!is.na(hads_wide_merged_all$SKN),] #remove missing stations with no SKN
  names(hads_wide_merged)[2]<-gsub("x.","X",names(hads_wide_merged)[2])#make lower case x to uppercase X for continuity 
  hads_wide_merged<-hads_wide_merged[!is.na(hads_wide_merged[,2]),] #remove NA rf day vals
  #tail(hads_wide_merged)
  count_log_hads<-data.frame(datastream=as.character("hads"),station_count=as.numeric(nrow(hads_wide_merged)),unique=as.logical(0)) #log of stations acquired
  print(paste(hads_month_filename,"found!",nrow(hads_wide_merged),"stations added!"))
  }else{ #else if nrow(hads_day) = 0 IE:no day data
   hads_wide_merged<-data.frame(sourceID=as.character(NA),date_day=as.numeric(NA),datastream=as.character("hads"),SKN=as.numeric(NA))
   names(hads_wide_merged)[2]<-format(Sys.Date()-1,"X%Y.%m.%d")
   missing_hads<-data.frame(sourceID=as.character(NA),datastream=as.character(NA)) #missing stations blank df
   count_log_hads<-data.frame(datastream=as.character("hads"),station_count=as.numeric(0),unique=as.logical(0))#log of stations acquired
   print(paste("NO HADS DATA:",map_date,"!"))
   }}else{ #else if hads month df is missing make a blank df
    hads_wide_merged<-data.frame(sourceID=as.character(NA),date_day=as.numeric(NA),datastream=as.character("hads"),SKN=as.numeric(NA))
    names(hads_wide_merged)[2]<-format(Sys.Date()-1,"X%Y.%m.%d")
	  missing_hads<-data.frame(sourceID=as.character(NA),datastream=as.character(NA)) #missing stations blank df
	  count_log_hads<-data.frame(datastream=as.character("hads"),station_count=as.numeric(0),unique=as.logical(0))#log of stations acquired
    print(paste(hads_month_filename,"MISSING empty DF made!"))
    }

#add NWS data 
setwd(nws_daily_wd)#set data source wd
nws_month_filename<-paste0(file_date,"_nws_daily_rf.csv")#dynamic file name that includes month year so when month is done new file is written
if(max(as.numeric(list.files()==nws_month_filename))>0){
 nws<-read.csv(nws_month_filename,header=T)
 nws<-nws[,c("nwsli","date","prec_mm_24hr")]
 names(nws)<-c("sourceID","date","x")
 nws$date<-as.Date(nws$date)#format as date
 nws_day<-nws[nws$date==map_date,]
 if(nrow(nws_day)>0){ #if nws_day has rows/data
  nws_day$date<-format(nws_day$date,"%Y.%m.%d")
  nws_day$x<-as.numeric(nws_day$x)
  nws_wide<- reshape(nws_day, idvar = "sourceID", timevar = "date", direction = "wide")
  nws_wide$datastream<-"nws"
  #head(nws_wide)
  #tail(nws_wide)
  nws_wide_merged_all<-merge(nws_wide,geog_meta_sub[,c("NWS.id","SKN")],by.x="sourceID",by.y="NWS.id",all.x=T)
  missing_nws<-nws_wide_merged_all[is.na(nws_wide_merged_all$SKN),c("sourceID","datastream")] #missing stations
  nws_wide_merged<-nws_wide_merged_all[!is.na(nws_wide_merged_all$SKN),] #remove missing stations with no SKN
  names(nws_wide_merged)[2]<-gsub("x.","X",names(nws_wide_merged)[2])#make lower case x to uppercase X for continuity
  nws_wide_merged<-nws_wide_merged[!is.na(nws_wide_merged[,2]),] #remove NA rf day vals
  #tail(nws_wide_merged)
  count_log_nws<-data.frame(datastream=as.character("nws"),station_count=as.numeric(nrow(nws_wide_merged)),unique=as.logical(0))
  print(paste(nws_month_filename,"found!",nrow(nws_wide_merged),"stations added!"))
  }else{ #else if nrow(nws_day) = 0
   nws_wide_merged<-data.frame(sourceID=as.character(NA),date_day=as.numeric(NA),datastream=as.character("nws"),SKN=as.numeric(NA))
   names(nws_wide_merged)[2]<-format(Sys.Date()-1,"X%Y.%m.%d")
   missing_nws<-data.frame(sourceID=as.character(NA),datastream=as.character(NA)) #missing stations blank df
   count_log_nws<-data.frame(datastream=as.character("nws"),station_count=as.numeric(0),unique=as.logical(0))
   print(paste("NO NWS DATA:",map_date,"!"))
   }}else{ #else if nws month df is missing make a blank df
    nws_wide_merged<-data.frame(sourceID=as.character(NA),date_day=as.numeric(NA),datastream=as.character("nws"),SKN=as.numeric(NA))
    names(nws_wide_merged)[2]<-format(Sys.Date()-1,"X%Y.%m.%d")
	  missing_nws<-data.frame(sourceID=as.character(NA),datastream=as.character(NA)) #missing stations blank df
	  count_log_nws<-data.frame(datastream=as.character("nws"),station_count=as.numeric(0),unique=as.logical(0))
    print(paste(nws_month_filename,"MISSING empty DF made!"))
    }

#add SCAN data
setwd(scan_daily_wd)#set data source wd
scan_month_filename<-paste0(file_date,"_scan_daily_rf.csv")
if(max(as.numeric(list.files()==scan_month_filename))>0){
 scan<-read.csv(scan_month_filename,header=T)
 #subset 24hr obs
 names(scan)<-c("sourceID","date","x")
 scan$date<-as.Date(scan$date)
 scan_day<- scan[scan$date==map_date,]
 if(nrow(scan_day)>0){ #if scan_day has rows/data
  scan_day$date<-format(scan_day$date,"%Y.%m.%d")
  scan_day$x<-as.numeric(scan_day$x)
  #head(scan_day)
  scan_wide<- reshape(scan_day, idvar = "sourceID", timevar = "date", direction = "wide")
  scan_wide$datastream<-"scan"
  #head(scan_wide)
  scan_wide_merged<-merge(scan_wide,geog_meta_sub[,c("SKN","SCAN.id")],by.x="sourceID",by.y="SCAN.id",all.x=T)
  missing_scan<- scan_wide_merged[is.na(scan_wide_merged$SKN),c("sourceID","datastream")] #missing stations
  scan_wide_merged<- scan_wide_merged[!is.na( scan_wide_merged$SKN),] #remove missing stations with no SKN
  names(scan_wide_merged)[2]<-gsub("x.","X",names(scan_wide_merged)[2])#make lower case x to uppercase X for continuity
  scan_wide_merged<-scan_wide_merged[!is.na(scan_wide_merged[,2]),] #remove NA rf day vals
  #tail(scan_wide_merged)
  count_log_scan<-data.frame(datastream=as.character("scan"),station_count=as.numeric(nrow(scan_wide_merged)),unique=as.logical(0))
  print(paste(scan_month_filename,"found!",nrow(scan_wide_merged),"stations added!"))
  }else{ #else if nrow(scan_day) = 0
   scan_wide_merged<-data.frame(sourceID=as.character(NA),date_day=as.numeric(NA),datastream=as.character("scan"),SKN=as.numeric(NA))
   names(scan_wide_merged)[2]<-format(Sys.Date()-1,"X%Y.%m.%d")
   missing_scan<-data.frame(sourceID=as.character(NA),datastream=as.character(NA)) #missing stations blank df
   count_log_scan<-data.frame(datastream=as.character("scan"),station_count=as.numeric(0),unique=as.logical(0))
   print(paste("NO SCAN DATA:",map_date,"!"))
   }}else{ #else if scan month df is missing make a blank df
    scan_wide_merged<-data.frame(sourceID=as.character(NA),date_day=as.numeric(NA),datastream=as.character("scan"),SKN=as.numeric(NA))
    names(scan_wide_merged)[2]<-format(Sys.Date()-1,"X%Y.%m.%d")
	  missing_scan<-data.frame(sourceID=as.character(NA),datastream=as.character(NA)) #missing stations blank df
    count_log_scan<-data.frame(datastream=as.character("scan"),station_count=as.numeric(0),unique=as.logical(0))
	  print(paste(scan_month_filename,"MISSING empty DF made!"))
    }

#make and write table of all missing stations from acquired data
all_missing<-rbind(missing_hads,missing_nws,missing_scan)
all_missing<-all_missing[!is.na(all_missing$sourceID),] #remove no sourceID stations
all_missing$lastDate<-as.Date(map_date)
setwd(missing_sta_wd) #set output wd for missing station
missing_files<-list.files()
missing_month_filename<-paste0(file_date,"_unknown_rf_sta.csv") #dynamic file name that includes month year so when month is done new file is written

#conditional statement that adds obs of missing stations and removes duplicate for the month
if(max(as.numeric(missing_files==missing_month_filename))>0){
	rf_missing_df<-read.csv(missing_month_filename)
	rf_missing_df$lastDate<-as.Date(rf_missing_df$lastDate)
	rf_missing_df<-rbind(rf_missing_df,all_missing)
	rf_missing_df<-rf_missing_df[order(rf_missing_df$lastDate, decreasing = TRUE),]
	rf_missing_df_nodups<-rf_missing_df[!duplicated(rf_missing_df$sourceID),]
	rf_missing_df_nodups<-rf_missing_df_nodups[order(rf_missing_df_nodups$lastDate),]
	write.csv(rf_missing_df_nodups,missing_month_filename, row.names=F)
	print("monthly unknown sta table appended!")
	print(paste(missing_month_filename,"unknown sta table appended!"))
	}else{
	write.csv(all_missing,missing_month_filename, row.names=F)
	print(paste(missing_month_filename,"unknown sta table written!"))
	}
print("unknown station table below...")
print(all_missing)

#rbind all data streams and merge with geog meta
print("combinding all data...")
hads_nws_wide<-rbind.all.columns(hads_wide_merged,nws_wide_merged)
hads_nws_scan_wide<-rbind.all.columns(hads_nws_wide,scan_wide_merged)
print("all data combind!")

#remove stations with NA values
rf_col<-paste0("X",format(map_date,"%Y.%m.%d"))#define rf day col name
hads_nws_scan_wide<-hads_nws_scan_wide[!is.na(hads_nws_scan_wide[,rf_col]),] #remove na rf obs should be none

#reorder to define data stream priority
data_priority <- c("hads","nws","scan")
hads_nws_scan_wide<-hads_nws_scan_wide[order(match(hads_nws_scan_wide$datastream, data_priority)),] #remove dup stations by priority
dim(hads_nws_scan_wide)
str(hads_nws_scan_wide)
head(hads_nws_scan_wide)
tail(hads_nws_scan_wide)
print("combind data sorted!")

#remove dup stations based on SKN but keeping data in order defined above
hads_nws_scan_wide_no_dup<-hads_nws_scan_wide[!duplicated(hads_nws_scan_wide$SKN), ]
str(hads_nws_scan_wide_no_dup)
print(paste("station count with dups:",nrow(hads_nws_scan_wide)))#~258+
print(paste("station count without dups:",nrow(hads_nws_scan_wide_no_dup)))#~159+

#sub cols rainfall
hads_nws_scan_wide_no_dup_rf<-hads_nws_scan_wide_no_dup[,c("SKN",rf_col)]#remove rf meta cols except SKN & RF DAY col
head(hads_nws_scan_wide_no_dup_rf)

#make rainfall source table and sub cols
hads_nws_scan_wide_no_dup_source<-hads_nws_scan_wide_no_dup[,c("SKN","datastream")]#remove meta cols except SKN & data stream cols
names(hads_nws_scan_wide_no_dup_source)<-names(hads_nws_scan_wide_no_dup_rf)#rename cols so source is date
head(hads_nws_scan_wide_no_dup_source)

#make and write table of source log station counts from acquired data
count_log_per<-rbind(count_log_hads,count_log_nws,count_log_scan)
count_log_unq<-data.frame(table(hads_nws_scan_wide_no_dup$datastream))
names(count_log_unq)<-c("datastream","station_count")
count_log_unq$unique<-as.logical(1)
count_log_all<-rbind(count_log_per,
				data.frame(datastream="SUBTOTAL",station_count=sum(count_log_per$station_count),unique=as.logical(0)),
				count_log_unq,
				data.frame(datastream="TOTAL",station_count=sum(count_log_unq$station_count),unique=as.logical(1)))
count_log_all$date<-map_date #add data date 
setwd(count_log_wd)
count_log_month_filename<-paste0(file_date,"_count_log_daily_rf.csv")#dynamic file name that includes month year so when month is done new file is written
count_files<-list.files()
	
#conditional statement that adds obs of per day station counts
if(max(as.numeric(count_files==count_log_month_filename))>0){
	write.table(count_log_all,count_log_month_filename, row.names=F,sep = ",",col.names = F, append = T)
	print(paste(count_log_month_filename,"daily station count appended!"))
	}else{
	write.csv(count_log_all,count_log_month_filename, row.names=F)
	print(paste(count_log_month_filename,"daily station count written!"))
	}
print("final station count table below...")
print(count_log_all)

#write or append daily source data
setwd(rf_day_source_wd)#set rainfall output wd
source_files<-list.files()
source_month_filename<-paste0("Statewide_Daily_Source_",file_date,".csv") #dynamic file name that includes month year so when month is done new file is written

#conditional statement that adds new obs
if(max(as.numeric(source_files==source_month_filename))>0){
  source_month_df<-read.csv(source_month_filename)
  sub_cols<-c("SKN",names(source_month_df)[grep("X",names(source_month_df))])
  final_source_data<-merge(source_month_df[,sub_cols],hads_nws_scan_wide_no_dup_source,by="SKN",all=T)
  final_source_data<-merge(geog_meta,final_source_data,by="SKN")
  write.csv(final_source_data,source_month_filename, row.names=F)
  print(paste(source_month_filename,"daily souce table appended!"))
}else{ #if month year file does not exist make a new month year file
  final_source_data<-merge(geog_meta,hads_nws_scan_wide_no_dup_source,by="SKN")
  write.csv(final_source_data,source_month_filename, row.names=F)
  print(paste(source_month_filename,"daily souce table written!"))
}

print("final source data table below...")
head(final_source_data)
tail(final_source_data)

#write data by creating or appending day to month for rf and source tables
#write or append daily rf data
setwd(rf_day_data_wd) #set rainfall output wd
rf_files<-list.files()
rf_month_filename<-paste0("Statewide_Raw_Daily_RF_mm_",file_date,".csv") #dynamic file name that includes month year so when month is done new file is writen

#conditional statement that adds new obs day col
if(max(as.numeric(rf_files==rf_month_filename))>0){
	rf_month_df<-read.csv(rf_month_filename)
	sub_cols<-c("SKN",names(rf_month_df)[grep("X",names(rf_month_df))])
	add_rf_data_sub<-merge(rf_month_df[,sub_cols],hads_nws_scan_wide_no_dup_rf,by="SKN",all=T)
	final_rf_data<-merge(geog_meta,add_rf_data_sub,by="SKN")
	write.csv(final_rf_data,rf_month_filename, row.names=F)
	print(paste(rf_month_filename,"daily rainfall table appended!"))
    }else{ #if month year file does not exist make a new month year file
	final_rf_data<-merge(geog_meta,hads_nws_scan_wide_no_dup_rf,by="SKN")
	write.csv(final_rf_data,rf_month_filename, row.names=F)
	print(paste(rf_month_filename,"daily rainfall table written!"))
}

print("final data table below...")
head(final_rf_data)
tail(final_rf_data)

paste(Sys.Date()-1,"DATA COMBIND RUN - CODE PAU!")

