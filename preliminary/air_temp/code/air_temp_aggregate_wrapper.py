import sys
import pandas as pd
from datetime import date
from temp_aggregate_input import aggregate_input

#DEFINE CONSTANTS-------------------------------------------------------------
MASTER_DIR = r'/home/kodamak8/share/air_temp/'
WORKING_MASTER_DIR = MASTER_DIR + r'working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'preliminary_output/'
PROC_DATA_DIR = WORKING_MASTER_DIR + r'processed_data/'
AGG_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw/'
META_MASTER_DIR = WORKING_MASTER_DIR + r'static_master_meta/'
META_MASTER_FILE = META_MASTER_DIR + r'Master_Sta_List_Meta_2021_07_19.csv'

TMIN_VARNAME = 'Tmin'
TMAX_VARNAME = 'Tmax'
SOURCE_LIST = ['hads']
#END CONSTANTS----------------------------------------------------------------

#Go through each source and aggregate everything in list
#Eventually consider staggered aggregation
date_str = date.today().strftime('%Y-%m-%d')
for src in SOURCE_LIST:
    year = date_str.split('-')[0]
    mon = date_str.split('-')[1]
    proc_tmin_file_name = '_'.join((TMIN_VARNAME,src,year,mon,'processed')) + '.csv'
    proc_tmax_file_name = '_'.join((TMAX_VARNAME,src,year,mon,'processed')) + '.csv'
    source_processed_dir = PROC_DATA_DIR + src + '/'
    aggregate_input(TMIN_VARNAME,proc_tmin_file_name,source_processed_dir,AGG_OUTPUT_DIR,META_MASTER_FILE)
    aggregate_input(TMAX_VARNAME,proc_tmax_file_name,source_processed_dir,AGG_OUTPUT_DIR,META_MASTER_FILE)
