import sys
import pytz
import pandas as pd
from datetime import datetime, timedelta
from monthly_cross_validate import main_monthly_cv,main_tmean_monthly_cv,get_metrics,write_meta_text
#DEFINE CONSTANTS-------------------------------------------------------------
TMAX_VARNAME = 'Tmax'
TMIN_VARNAME = 'Tmin'
#END CONSTANTS----------------------------------------------------------------

if __name__=="__main__":
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        date_time = pd.to_datetime(date_str)
        this_year = date_time.year
        this_mon = date_time.month
        month_st = datetime(this_year,this_mon,1)
    else:
        today = datetime.today()
        this_year = today.year
        this_mon = today.month
        this_st = datetime(this_year,this_mon,1)
        prev = this_st - timedelta(days=1)
        prev_year = prev.year
        prev_mon = prev.month
        month_st = datetime(prev_year,prev_mon,1)

    month_date = month_st.strftime('%Y-%m-%d')
    icode_list = ['BI','KA','MN','OA']
    version_type = 'preliminary'

    #Tmin and Tmax first
    for icode in icode_list:
        try:
            mode = 'max'
            cv_data = main_monthly_cv(TMAX_VARNAME,month_date,icode)
            meta = get_metrics(TMAX_VARNAME,icode,month_date)
            meta_file = write_meta_text(mode,month_date,meta,version_type)
            print(TMAX_VARNAME,icode,month_date,'done')
        except:
            print(TMAX_VARNAME,icode,month_date,'failed')
        try:
            mode = 'min'
            cv_data = main_monthly_cv(TMIN_VARNAME,month_date,icode)
            meta = get_metrics(TMIN_VARNAME,icode,month_date)
            meta_file = write_meta_text(mode,month_date,meta,version_type)
            print(TMIN_VARNAME,icode,month_date,'done')
        except:
            print(TMIN_VARNAME,icode,month_date,'failed')

        try:
            mode = 'mean'
            cv_data = main_tmean_monthly_cv(month_date,icode)
            meta = get_metrics('Tmean',icode,month_date)
            meta_file = write_meta_text(mode,month_date,meta,version_type)
            print('Tmean',icode,month_date,'done')
        except:
            print('Tmean',icode,month_date,'failed')