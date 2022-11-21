"""
Prior to temperature daily data aggregation, run this to pull requisite files, if exist
Pull
-yyyymmdd_hads_parsed.csv
-yyyymmdd_madis_parsed.csv
-daily_Tmin_yyyy_mm.csv (if exist)
-daily_Tmax_yyyy_mm.csv (if exist)
"""
import sys
import subprocess
import pytz
import pandas as pd
from datetime import datetime, timedelta

PARENT_DIR = r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/workflow_data/preliminary_test/'
REMOTE_BASEURL =r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/production/temperature/'
LOCAL_PARENT = r'/home/hawaii_climate_products_container/preliminary/'
LOCAL_DATA_AQS = LOCAL_PARENT + r'data_aqs/data_outputs/'
LOCAL_TEMP = LOCAL_PARENT + r'air_temp/data_outputs/tables/station_data/daily/raw/statewide/'
SRC_LIST = ['hads','madis']

if __name__=='__main__':
    if len(sys.argv) > 1:
        input_date = sys.argv[1]
        dt = pd.to_datetime(input_date)
        prev_day_day = dt.strftime('%Y%m%d')
        prev_day_mon = dt.strftime('%Y_%m')
        year_str = dt.strftime('%Y')
        mon_str = dt.strftime('%m')
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        prev_day = today - timedelta(days=1)
        prev_day_day = prev_day.strftime('%Y%m%d')
        prev_day_mon = prev_day.strftime('%Y_%m')
        year_str = prev_day.strftime('%Y')
        mon_str = prev_day.strftime('%m')

    #Pull the daily data acquisitions
    for src in SRC_LIST:
        src_url = PARENT_DIR+r'data_aqs/data_outputs/'+src+r'/parse/'
        dest_url = LOCAL_DATA_AQS + src + r'/parse/'
        filename = src_url + r'_'.join((prev_day_day,src,'parsed')) + r'.csv'
        local_name = dest_url + r'_'.join((prev_day_day,src,'parsed')) + r'.csv'
        cmd = ["wget",filename,"-O",local_name]
        subprocess.call(cmd)
        
    #Tmin daily stations pull
    src_url = REMOTE_BASEURL + r'min/day/statewide/raw/station_data/'+year_str+r'/'+mon_str+r'/'
    filename = src_url + r'_'.join(('temperature','min','day_statewide_raw_station_data',prev_day_mon)) + r'.csv'
    local_name = LOCAL_TEMP + r'_'.join(('daily','Tmin',prev_day_mon)) + r'.csv'
    cmd = ["wget",filename,"-O",local_name]
    subprocess.call(cmd)

    #Tmax daily stations pull
    src_url = REMOTE_BASEURL + r'max/day/statewide/raw/station_data/'+year_str+r'/'+mon_str+r'/'
    filename = src_url + r'_'.join(('temperature','max','day_statewide_raw_station_data',prev_day_mon)) + r'.csv'
    local_name = LOCAL_TEMP + r'_'.join(('daily','Tmax',prev_day_mon)) + r'.csv'
    cmd = ["wget",filename,"-O",local_name]
    subprocess.call(cmd)




