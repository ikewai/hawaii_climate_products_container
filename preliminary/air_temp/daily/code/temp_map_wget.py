"""
Runs prior to mapping workflow
"""
import subprocess
import pytz
from datetime import datetime, timedelta

PARENT_DIR = r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/workflow_data/preliminary_test/'
DEPEND_DIR = r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/temperature/'
LOCAL_PARENT = r'/home/hawaii_climate_products_container/preliminary/'
LOCAL_DEPEND = LOCAL_PARENT + r'air_temp/daily/'
LOCAL_TEMP = LOCAL_PARENT + r'air_temp/data_outputs/tables/station_data/daily/raw/statewide/'

hst = pytz.timezone('HST')
today = datetime.today().astimezone(hst)
prev_day = today - timedelta(days=1)
prev_day_day = prev_day.strftime('%Y%m%d')
prev_day_mon = prev_day.strftime('%Y_%m')

#Tmin daily stations pull
src_url = PARENT_DIR + r'air_temp/data_outputs/tables/station_data/daily/raw/statewide/'
filename = src_url + r'_'.join(('daily','Tmin',prev_day_mon)) + r'.csv'
cmd = ["wget",filename,"-P",LOCAL_TEMP]
subprocess.call(cmd)

#Tmax daily stations pull
filename = src_url + r'_'.join(('daily','Tmax',prev_day_mon)) + r'.csv'
cmd = ["wget",filename,"-P",LOCAL_TEMP]
subprocess.call(cmd)

#Air temp daily dependencies
src_url = DEPEND_DIR + "dependencies.tar.gz"
dest_path = LOCAL_DEPEND + "dependencies.tar.gz"
cmd = ["wget",src_url,"-O",dest_path]
subprocess.call(cmd)
cmd = ["tar","-xvf",dest_path]
subprocess.call(cmd)
