import sys
import numpy as np
from pandas import to_datetime
from datetime import date, timedelta
from temp_state_aggregate import statewide_mosaic, create_tables, qc_state_aggregate

#DEFINE CONSTANTS-------------------------------------------------------------
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
CODE_MASTER_DIR = MASTER_DIR + r'air_temp/daily/code/'
DEP_MASTER_DIR = MASTER_DIR + r'air_temp/daily/dependencies/'
RUN_MASTER_DIR = MASTER_DIR + r'air_temp/data_outputs/'
COUNTY_MAP_DIR = RUN_MASTER_DIR + r'tiffs/daily/county/' #Set subdirectories based on varname and iCode
STATE_MAP_DIR = RUN_MASTER_DIR + r'tiffs/daily/statewide/'
SE_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/daily/county/'
CV_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/loocv/daily/county/'
ICODE_LIST = ['BI','KA','MN','OA']
TEMP_SUFF = ''
SE_SUFF = '_se'
#END CONSTANTS----------------------------------------------------------------

if __name__=="__main__":
    if len(sys.argv) > 1:
        input_date = sys.argv[1]
        dt = to_datetime(input_date)
        date_str = dt.strftime('%Y-%m-%d')
    else:
        today = date.today()
        prev_day = today - timedelta(days=1)
        date_str = prev_day.strftime('%Y-%m-%d')
    print(date_str)
    #Tmin section
    varname = 'Tmin'
    #QC station data
    tmin_state_qc = qc_state_aggregate(varname,date_str)
    #Maps
    statewide_mosaic(varname,date_str,COUNTY_MAP_DIR,TEMP_SUFF,STATE_MAP_DIR)
    #SE maps
    statewide_mosaic(varname,date_str,COUNTY_MAP_DIR,SE_SUFF,STATE_MAP_DIR)
    #Meta data
    create_tables('T','min',date_str)

    #Tmax section
    varname = 'Tmax'
    #QC station data
    tmax_state_qc = qc_state_aggregate(varname,date_str)
    #Maps
    statewide_mosaic(varname,date_str,COUNTY_MAP_DIR,TEMP_SUFF,STATE_MAP_DIR)
    #SE maps
    statewide_mosaic(varname,date_str,COUNTY_MAP_DIR,SE_SUFF,STATE_MAP_DIR)
    #Meta data
    create_tables('T','max',date_str)

    #Tmean section
    varname = 'Tmean'
    #Maps
    statewide_mosaic(varname,date_str,COUNTY_MAP_DIR,TEMP_SUFF,STATE_MAP_DIR)
    #SE maps
    statewide_mosaic(varname,date_str,COUNTY_MAP_DIR,SE_SUFF,STATE_MAP_DIR)
    #Meta data
    create_tables('T','mean',date_str)

