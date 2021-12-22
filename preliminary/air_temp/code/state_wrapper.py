import numpy as np

from datetime import date, timedelta
from temp_state_aggregate import statewide_mosaic, create_tables, qc_state_aggregate

#DEFINE CONSTANTS-------------------------------------------------------------
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/air_temp/'
CODE_MASTER_DIR = MASTER_DIR + r'code/'
WORKING_MASTER_DIR = MASTER_DIR + r'working_data/'
RUN_MASTER_DIR = MASTER_DIR + r'data_outputs/'
COUNTY_MAP_DIR = RUN_MASTER_DIR + r'tiffs/daily/county/' #Set subdirectories based on varname and iCode
STATE_MAP_DIR = RUN_MASTER_DIR + r'tiffs/daily/statewide/'
SE_OUTPUT_DIR = RUN_MASTER_DIR + r'tiffs/daily/county/'
CV_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/loocv/daily/county/'
ICODE_LIST = ['BI','KA','MN','OA']
TEMP_SUFF = ''
SE_SUFF = '_se'
#END CONSTANTS----------------------------------------------------------------

today = date.today()
prev_day = today - timedelta(days=1)
date_str = prev_day.strftime('%Y-%m-%d')

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

