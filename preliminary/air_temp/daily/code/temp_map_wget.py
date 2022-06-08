"""
Runs prior to mapping workflow
"""
import sys
import subprocess
import pytz
from datetime import datetime, timedelta
from pandas import to_datetime

REMOTE_BASEURL =r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/production/temperature/'
DEPEND_DIR = r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/temperature/'
LOCAL_PARENT = r'/home/hawaii_climate_products_container/preliminary/'
LOCAL_DEPEND = LOCAL_PARENT + r'air_temp/daily/'
LOCAL_TEMP = LOCAL_PARENT + r'air_temp/data_outputs/tables/station_data/daily/raw/statewide/'

if __name__=="__main__":
    if len(sys.argv) > 1:
        input_date = sys.argv[1]
        dt = to_datetime(input_date)
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

    #Tmin daily stations pull
    src_url = REMOTE_BASEURL + r'min/day/statewide/raw/station_data/'+year_str+r'/'+mon_str+r'/'
    filename = src_url + r'_'.join(('daily','Tmin',prev_day_mon)) + r'.csv'
    local_name = LOCAL_TEMP + r'_'.join(('daily','Tmin',prev_day_mon)) + r'.csv'
    cmd = ["wget",filename,"-O",local_name]
    #subprocess.call(cmd)

    #Tmax daily stations pull
    src_url = REMOTE_BASEURL + r'max/day/statewide/raw/station_data/'+year_str+r'/'+mon_str+r'/'
    filename = src_url + r'_'.join(('daily','Tmax',prev_day_mon)) + r'.csv'
    local_name = LOCAL_TEMP + r'_'.join(('daily','Tmax',prev_day_mon)) + r'.csv'
    cmd = ["wget",filename,"-O",local_name]
    #subprocess.call(cmd)

    #Air temp daily dependencies
    src_url = DEPEND_DIR + "dependencies.tar.gz"
    dest_path = LOCAL_DEPEND + "dependencies.tar.gz"
    cmd = ["wget",src_url,"-O",dest_path]
    subprocess.call(cmd)
    cmd = ["tar","-xvf",dest_path]
    subprocess.call(cmd)
