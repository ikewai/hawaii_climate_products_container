import pytz
from datetime import datetime, timedelta
from daily_temp_map_batch import generate_county_map
from daily_temp_mean_batch import generate_county_mean, generate_se_mean

#DEFINE CONSTANTS-------------------------------------------------------------

ICODE_LIST = ['BI','KA','MN','OA']
PARAM_LIST = ['dem_250']
#END CONSTANTS----------------------------------------------------------------

#Set date to previous 24 hours
hst = pytz.timezone('HST')
today = datetime.today().astimezone(hst)
prev_day = today - timedelta(days=1)
date_str = prev_day.strftime('%Y-%m-%d')
print(date_str)
#Tmin county maps
varname = 'Tmin'
for icode in ICODE_LIST:
    temp_df,pred_df,theta,isl_dims = generate_county_map(icode,varname,PARAM_LIST,date_str)
    print(varname,icode,'complete')
#Tmax county maps
varname = 'Tmax'
for icode in ICODE_LIST:
    temp_df,pred_df,theta,isl_dims = generate_county_map(icode,varname,PARAM_LIST,date_str)
    print(varname,icode,'complete')

varname = 'Tmean'
for icode in ICODE_LIST:
    generate_county_mean(icode,date_str)
    generate_se_mean(icode,date_str)
    print(varname,icode,'complete')