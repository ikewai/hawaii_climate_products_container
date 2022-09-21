import sys
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from monthly_temp_maps import minmax_main, mean_main

#DEFINE CONSTANTS-------------------------------------------------------------
TMAX_VARNAME = 'Tmax'
TMIN_VARNAME = 'Tmin'
#END CONSTANTS----------------------------------------------------------------

if __name__=="__main__":
    if len(sys.argv) > 1:
        #If manual input of date
        date_str = sys.argv[1]
        date_time = pd.to_datetime(date_str)
        this_year = date_time.year
        this_mon = date_time.month
        month_st = datetime(this_year,this_mon,1)
        #Double check that the month is complete
        #If manual start is the same as today month start, month not complete
        today = datetime.today()
        today_year = today.year
        today_mon = today.month
        today_st = datetime(today_year,today_mon,1)
        if month_st == today_st:
            print('Month incomplete. Exiting.')
            quit()
    else:
        #Automatic real-time date set
        #Runs for the last complete month prior to today()
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
    #Run for tmax and tmin first
    for icode in icode_list:
        try:
            T_model,se_model,theta,pcov = minmax_main(TMAX_VARNAME,icode,month_date)
            print(TMAX_VARNAME,icode,month_date,'done')
        except:
            print(TMAX_VARNAME,icode,month_date,'failed')
        try:
            T_model,se_model,theta,pcov = minmax_main(TMIN_VARNAME,icode,month_date)
            print(TMIN_VARNAME,icode,month_date,'done')
        except:
            print(TMIN_VARNAME,icode,month_date,'failed')
        try:
            Tmean,se_mean = mean_main(icode,month_date)
            print('Tmean',icode,month_date,'done')
        except:
            print('Tmean',icode,month_date,'failed')



