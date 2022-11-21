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
LOCAL_DEPEND = LOCAL_PARENT + r'air_temp/monthly/'
LOCAL_TEMP = LOCAL_PARENT + r'air_temp/data_outputs/tables/station_data/daily/raw_qc/statewide/'

if __name__=="__main__":
    if len(sys.argv) > 1:
        input_date = sys.argv[1]
        dt = to_datetime(input_date)
        this_year = dt.year
        this_mon = dt.month
        month_st = datetime(this_year,this_mon,1)
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        today_year = today.year
        today_mon = today.month
        today_st = datetime(today_year,today_mon,1)
        if month_st == today_st:
            print('Month incomplete. Exiting.')
            quit()
        else:
            year_str = month_st.strftime('%Y')
            mon_str = month_st.strftime('%m')
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        this_year = today.year
        this_mon = today.month
        this_st = datetime(this_year,this_mon,1)
        prev = this_st - timedelta(days=1)
        prev_year = prev.year
        prev_mon = prev.month
        month_st = datetime(prev_year,prev_mon,1)
        year_str = month_st.strftime('%Y')
        mon_str = month_st.strftime('%m')

    #Tmin daily stations pull
    src_url = REMOTE_BASEURL + r'min/day/statewide/partial/station_data/'+year_str+r'/'+mon_str+r'/'
    filename = src_url + r'_'.join(('temperature','min','day_statewide_partial_station_data',year_str,mon_str)) + r'.csv'
    local_name = LOCAL_TEMP + r'_'.join(('daily','Tmin',year_str,mon_str,'qc')) + r'.csv'
    cmd = ["wget",filename,"-O",local_name]
    subprocess.call(cmd)

    #Tmax daily stations pull
    src_url = REMOTE_BASEURL + r'max/day/statewide/partial/station_data/'+year_str+r'/'+mon_str+r'/'
    filename = src_url + r'_'.join(('temperature','max','day_statewide_partial_station_data',year_str,mon_str)) + r'.csv'
    local_name = LOCAL_TEMP + r'_'.join(('daily','Tmax',year_str,mon_str,'qc')) + r'.csv'
    cmd = ["wget",filename,"-O",local_name]
    subprocess.call(cmd)

    #Air temp daily dependencies
    src_url = DEPEND_DIR + "dependencies.tar.gz"
    dest_path = LOCAL_DEPEND + "dependencies.tar.gz"
    cmd = ["wget",src_url,"-O",dest_path]
    subprocess.call(cmd)
    cmd = ["tar","-xvf",dest_path,"-C",LOCAL_DEPEND]
    subprocess.call(cmd)
