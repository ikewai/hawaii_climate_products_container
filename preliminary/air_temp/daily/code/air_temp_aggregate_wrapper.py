import sys
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from os.path import exists
from temp_aggregate_input import aggregate_input

#DEFINE CONSTANTS-------------------------------------------------------------
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
WORKING_MASTER_DIR = MASTER_DIR + r'air_temp/working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'air_temp/data_outputs/'
PROC_DATA_DIR = WORKING_MASTER_DIR + r'processed_data/'
AGG_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw/statewide/'
TRACK_DIR = RUN_MASTER_DIR + r'tables/air_temp_station_tracking/'
META_MASTER_FILE = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'

TMIN_VARNAME = 'Tmin'
TMAX_VARNAME = 'Tmax'
SOURCE_LIST = ['madis','hads']
COUNT_LIST = ['hads','madis']
#END CONSTANTS----------------------------------------------------------------

#DEFINE FUNCTIONS-------------------------------------------------------------
def get_daily_count(count_file,stn_list,date_str):
    count_cols = ['datastream','station_count','unique','date']
    non_unique_df = pd.DataFrame(columns=count_cols)
    unique_df = pd.DataFrame(columns=count_cols)
    prev_stations = []
    for src in COUNT_LIST:
        src_count = len(stn_list[src])
        unique_list = np.setdiff1d(list(stn_list[src]),prev_stations)
        unique_count = len(unique_list)
        non_unique_df.loc[len(non_unique_df)] = [src,src_count,'False',date_str]
        unique_df.loc[len(unique_df)] = [src,unique_count,'True',date_str]
        prev_stations = prev_stations + list(stn_list[src])
    non_unique_df.loc[len(non_unique_df)] = ['SUBTOTAL',non_unique_df['station_count'].sum(),'False',date_str]
    unique_df.loc[len(unique_df)] = ['TOTAL',unique_df['station_count'].sum(),'True',date_str]
    count_df = pd.concat([non_unique_df,unique_df],axis=0,ignore_index=True)
    if exists(count_file):
        prev_count_df = pd.read_csv(count_file)
        upd_count_df = pd.concat([prev_count_df,count_df],axis=0,ignore_index=True)
        upd_count_df = upd_count_df.drop_duplicates()
        upd_count_df.to_csv(count_file,index=False)
    else:
        count_df.to_csv(count_file,index=False)
            
#END FUNCTIONS----------------------------------------------------------------

#Go through each source and aggregate everything in list
#Eventually consider staggered aggregation
hst = pytz.timezone('HST')
today = datetime.today().astimezone(hst)
prev_day = today - timedelta(days=1)
date_str = prev_day.strftime('%Y-%m-%d')
tmin_stns_by_src = {}
tmax_stns_by_src = {}
for src in SOURCE_LIST:
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    proc_tmin_file_name = '_'.join((TMIN_VARNAME,src,year,mon,'processed')) + '.csv'
    proc_tmax_file_name = '_'.join((TMAX_VARNAME,src,year,mon,'processed')) + '.csv'
    #[SET_DIR]
    source_processed_dir = PROC_DATA_DIR + src + '/'
    tmin_agg_df,tmin_stns = aggregate_input(TMIN_VARNAME,proc_tmin_file_name,source_processed_dir,AGG_OUTPUT_DIR,META_MASTER_FILE)
    tmax_agg_df,tmax_stns = aggregate_input(TMAX_VARNAME,proc_tmax_file_name,source_processed_dir,AGG_OUTPUT_DIR,META_MASTER_FILE)
    tmin_stns_by_src[src] = tmin_stns
    tmax_stns_by_src[src] = tmax_stns

tmin_count_file = TRACK_DIR + '_'.join(('count_log_daily_Tmin',year,mon)) + '.csv'
tmax_count_file = TRACK_DIR + '_'.join(('count_log_daily_Tmax',year,mon)) + '.csv'
get_daily_count(tmin_count_file,tmin_stns_by_src,date_str)
get_daily_count(tmax_count_file,tmax_stns_by_src,date_str)