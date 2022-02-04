import numpy as np
from datetime import date, timedelta
import Temp_linear as tmpl
import cross_validate_temp as cv

#DEFINE CONSTANTS-------------------------------------------------------------
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
WORKING_MASTER_DIR = MASTER_DIR + r'air_temp/working_data/'
DEP_MASTER_DIR = MASTER_DIR + r'air_temp/dependencies/'
RUN_MASTER_DIR = MASTER_DIR + r'air_temp/data_outputs/'
PRED_DIR = DEP_MASTER_DIR + r'predictors/'
RAW_DATA_DIR = RUN_MASTER_DIR + r'tables/station_data/daily/raw/statewide/' #Location of station and predictor data for model fit
CV_OUTPUT_DIR = RUN_MASTER_DIR + r'tables/loocv/daily/county/'
ICODE_LIST = ['BI','KA','MN','OA']
PARAM_LIST = ['dem_250']
#END CONSTANTS----------------------------------------------------------------
today = date.today()
prev_day = today - timedelta(days=1)
date_str = prev_day.strftime('%Y-%m-%d')
print(date_str)

date_year = date_str.split('-')[0]
date_mon = date_str.split('-')[1]
cv_dir = CV_OUTPUT_DIR
#Tmin section
varname = 'Tmin'
temp_file = RAW_DATA_DIR + '_'.join(('daily',varname,date_year,date_mon)) + '.csv'
pred_file = PRED_DIR + varname.lower() + '_predictors.csv'

temp_df,temp_meta,temp_data = tmpl.extract_temp_input(temp_file)
pred_df,pred_sr = tmpl.extract_predictors(pred_file,PARAM_LIST)

for icode in ICODE_LIST:
    temp_date = tmpl.get_temperature_date(temp_data,temp_meta,icode,date_str,varname=varname)
    valid_skns = np.intersect1d(temp_date.index.values,pred_sr.index.values)
    temp_date = temp_date.loc[valid_skns]
    pred_date = pred_sr.loc[valid_skns]
    cv_tmin = cv.main_cv_single(varname,date_str,temp_date,pred_date,icode,cv_dir)
    meta_stats = cv.get_metrics(varname,icode,date_str,PARAM_LIST)
    temp_text = cv.write_meta_text(varname,date_str,meta_stats)
    print(varname,icode,'done')

varname = 'Tmax'
temp_file = RAW_DATA_DIR + '_'.join(('daily',varname,date_year,date_mon)) + '.csv'
pred_file = PRED_DIR + varname.lower() + '_predictors.csv'

temp_df,temp_meta,temp_data = tmpl.extract_temp_input(temp_file)
pred_df,pred_sr = tmpl.extract_predictors(pred_file,PARAM_LIST)

for icode in ICODE_LIST:
    temp_date = tmpl.get_temperature_date(temp_data,temp_meta,icode,date_str,varname=varname)
    valid_skns = np.intersect1d(temp_date.index.values,pred_sr.index.values)
    temp_date = temp_date.loc[valid_skns]
    pred_date = pred_sr.loc[valid_skns]
    cv_tmax = cv.main_cv_single(varname,date_str,temp_date,pred_date,icode,cv_dir)
    meta_stats = cv.get_metrics(varname,icode,date_str,PARAM_LIST)
    temp_text = cv.write_meta_text(varname,date_str,meta_stats)
    print(varname,icode,'done')


varname = 'Tmean'
for icode in ICODE_LIST:
    cv_tmean = cv.main_cv_mean(date_str,icode,cv_dir)
    meta_stats = cv.get_metrics(varname,icode,date_str,PARAM_LIST)
    temp_text = cv.write_meta_text(varname,date_str,meta_stats)
    print(varname,icode,'done')

